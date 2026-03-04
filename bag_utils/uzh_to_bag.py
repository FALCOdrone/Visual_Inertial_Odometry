#!/usr/bin/env python3
"""
Convert UZH FPV indoor dataset to a ROS2 bag file (.db3).

Images are fisheye-undistorted at write time so the bag is directly compatible
with the VIO pipeline (which assumes a pinhole / zero-distortion model).

If the bag already exists, validates per-topic message counts and only
rebuilds topics that are missing or have incorrect counts.

Usage:
    python3 uzh_to_bag.py --dataset dataset/uzh_indoor_9 --output bags/uzh_indoor_9
"""

import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np

from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from builtin_interfaces.msg import Time
import rosbag2_py


# ── UZH indoor camera calibration (equidistant / fisheye model) ───────────────
#   Source: indoor_forward_calib_snapdragon/camchain-imucam-...yaml
CAM0_INTRINSICS = [278.66723066149086, 278.48991409740296, 319.75221200593535, 241.96858910358173]
CAM0_DISTORTION = np.array(
    [-0.013721808247486035, 0.020727425669427896, -0.012786476702685545, 0.0025242267320687625],
    dtype=np.float64,
)

CAM1_INTRINSICS = [277.61640629770613, 277.63749695723294, 314.8944703346039, 236.04310050462587]
CAM1_DISTORTION = np.array(
    [-0.008456929295619607, 0.011407590938612062, -0.006951788325762078, 0.0015368127092821786],
    dtype=np.float64,
)

RESOLUTION = (640, 480)  # width, height


# ── Topic definitions ──────────────────────────────────────────────────────────
ALL_TOPICS = [
    ("/cam0/image_raw", "sensor_msgs/msg/Image"),
    ("/cam1/image_raw", "sensor_msgs/msg/Image"),
    ("/cam0/camera_info", "sensor_msgs/msg/CameraInfo"),
    ("/cam1/camera_info", "sensor_msgs/msg/CameraInfo"),
    ("/imu0", "sensor_msgs/msg/Imu"),
    ("/gt/pose", "geometry_msgs/msg/PoseStamped"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _sec_to_ns(t_sec: float) -> int:
    """Convert float seconds to integer nanoseconds."""
    return int(round(t_sec * 1_000_000_000))


def progress_bar(current, total, label="", bar_width=40, start_time=None):
    frac = current / total if total > 0 else 1.0
    filled = int(bar_width * frac)
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
    pct = frac * 100

    eta_str = ""
    if start_time is not None and current > 0:
        elapsed = time.time() - start_time
        eta = elapsed / current * (total - current)
        if eta >= 60:
            eta_str = f"  ETA {int(eta // 60)}m{int(eta % 60):02d}s"
        else:
            eta_str = f"  ETA {eta:.0f}s"

    line = f"\r  {label} |{bar}| {current}/{total} ({pct:5.1f}%){eta_str}  "
    sys.stderr.write(line)
    sys.stderr.flush()
    if current >= total:
        sys.stderr.write("\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  UZH dataset readers
# ═══════════════════════════════════════════════════════════════════════════════


def read_image_txt(txt_path):
    """
    Read left_images.txt / right_images.txt.
    Format (space-separated): id  timestamp_sec  img/image_X_Y.png
    Returns list of (timestamp_ns: int, rel_image_path: str).
    """
    entries = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts_ns = _sec_to_ns(float(parts[1]))
            entries.append((ts_ns, parts[2]))
    return entries


def read_imu_txt(txt_path):
    """
    Read imu.txt.
    Format (space-separated): id  timestamp_sec  gx  gy  gz  ax  ay  az
    Returns list of (timestamp_ns, gx, gy, gz, ax, ay, az).
    """
    entries = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts_ns = _sec_to_ns(float(parts[1]))
            gx, gy, gz = float(parts[2]), float(parts[3]), float(parts[4])
            ax, ay, az = float(parts[5]), float(parts[6]), float(parts[7])
            entries.append((ts_ns, gx, gy, gz, ax, ay, az))
    return entries


def read_groundtruth_txt(txt_path):
    """
    Read groundtruth.txt.
    Format (space-separated): timestamp_sec  tx  ty  tz  qx  qy  qz  qw
    Returns list of (timestamp_ns, tx, ty, tz, qw, qx, qy, qz).
    Note: UZH stores qx qy qz qw; we reorder to (qw, qx, qy, qz) for internal
    use, matching make_pose_stamped() which takes (qw, qx, qy, qz).
    """
    entries = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ts_ns = _sec_to_ns(float(parts[0]))
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            entries.append((ts_ns, tx, ty, tz, qw, qx, qy, qz))
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
#  Message builders
# ═══════════════════════════════════════════════════════════════════════════════


def ns_to_time_msg(timestamp_ns: int) -> Time:
    t = Time()
    t.sec = int(timestamp_ns // 1_000_000_000)
    t.nanosec = int(timestamp_ns % 1_000_000_000)
    return t


def make_camera_info(intrinsics, resolution, timestamp_ns, frame_id):
    """CameraInfo for already-undistorted images (zero distortion, plumb_bob)."""
    ci = CameraInfo()
    ci.header.stamp = ns_to_time_msg(timestamp_ns)
    ci.header.frame_id = frame_id
    ci.width = resolution[0]
    ci.height = resolution[1]
    ci.distortion_model = "plumb_bob"
    ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    fx, fy, cx, cy = intrinsics
    ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return ci


def _build_undistort_maps(intrinsics, distortion_fisheye, resolution):
    """Pre-compute fisheye undistortion maps for a camera."""
    fx, fy, cx, cy = intrinsics
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = distortion_fisheye.reshape(-1, 1)
    w, h = resolution
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    return map1, map2


def make_image_msg(image_path, timestamp_ns, frame_id, map1, map2):
    """Load image, apply pre-computed fisheye undistortion maps, build Image msg."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    msg = Image()
    msg.header.stamp = ns_to_time_msg(timestamp_ns)
    msg.header.frame_id = frame_id
    msg.height = img_undist.shape[0]
    msg.width = img_undist.shape[1]
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.step = img_undist.shape[1]
    msg.data = img_undist.tobytes()
    return msg


def make_imu_msg(timestamp_ns, gx, gy, gz, ax, ay, az):
    msg = Imu()
    msg.header.stamp = ns_to_time_msg(timestamp_ns)
    msg.header.frame_id = "imu0"
    msg.angular_velocity.x = gx
    msg.angular_velocity.y = gy
    msg.angular_velocity.z = gz
    msg.linear_acceleration.x = ax
    msg.linear_acceleration.y = ay
    msg.linear_acceleration.z = az
    msg.angular_velocity_covariance = [0.0] * 9
    msg.linear_acceleration_covariance = [0.0] * 9
    msg.orientation_covariance = [-1.0] + [0.0] * 8
    return msg


def make_pose_stamped(timestamp_ns, px, py, pz, qw, qx, qy, qz, frame_id="world"):
    msg = PoseStamped()
    msg.header.stamp = ns_to_time_msg(timestamp_ns)
    msg.header.frame_id = frame_id
    msg.pose.position = Point(x=px, y=py, z=pz)
    msg.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
#  Bag validation
# ═══════════════════════════════════════════════════════════════════════════════


def read_bag_topic_counts(bag_path):
    try:
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_counts = {t.name: 0 for t in reader.get_all_topics_and_types()}

        total_read = 0
        t0 = time.time()
        while reader.has_next():
            topic, _, _ = reader.read_next()
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            total_read += 1
            if total_read % 1000 == 0:
                progress_bar(total_read, total_read, "scanning", start_time=t0)

        progress_bar(1, 1, "scanning", start_time=t0)
        del reader
        return topic_counts
    except Exception as e:
        print(f"  Could not read existing bag: {e}")
        return None


def compute_expected_counts(cam0_entries, cam1_entries, imu_entries, gt_entries):
    return {
        "/cam0/image_raw": len(cam0_entries),
        "/cam1/image_raw": len(cam1_entries),
        "/cam0/camera_info": len(cam0_entries),
        "/cam1/camera_info": len(cam1_entries),
        "/imu0": len(imu_entries),
        "/gt/pose": len(gt_entries),
    }


def validate_bag(bag_path, expected):
    actual = read_bag_topic_counts(bag_path)
    if actual is None:
        return False, {t: (0, expected[t]) for t in expected}

    problems = {}
    for topic, exp_count in expected.items():
        bag_count = actual.get(topic, 0)
        if bag_count != exp_count:
            problems[topic] = (bag_count, exp_count)

    return len(problems) == 0, problems


# ═══════════════════════════════════════════════════════════════════════════════
#  Bag writing helpers
# ═══════════════════════════════════════════════════════════════════════════════


def write_imu(writer, imu_entries):
    t0 = time.time()
    for i, (ts_ns, gx, gy, gz, ax, ay, az) in enumerate(imu_entries):
        msg = make_imu_msg(ts_ns, gx, gy, gz, ax, ay, az)
        writer.write("/imu0", serialize_message(msg), ts_ns)
        if (i + 1) % 500 == 0 or i + 1 == len(imu_entries):
            progress_bar(i + 1, len(imu_entries), "/imu0", start_time=t0)


def write_cam(writer, cam_entries, dataset_dir, cam_name, intrinsics, map1, map2):
    img_topic = f"/{cam_name}/image_raw"
    info_topic = f"/{cam_name}/camera_info"
    t0 = time.time()
    for i, (ts_ns, rel_path) in enumerate(cam_entries):
        img_path = os.path.join(dataset_dir, rel_path)
        img_msg = make_image_msg(img_path, ts_ns, cam_name, map1, map2)
        if img_msg is None:
            sys.stderr.write(f"\n    WARNING: Could not read {img_path}\n")
            continue
        writer.write(img_topic, serialize_message(img_msg), ts_ns)
        ci_msg = make_camera_info(intrinsics, RESOLUTION, ts_ns, cam_name)
        writer.write(info_topic, serialize_message(ci_msg), ts_ns)
        if (i + 1) % 50 == 0 or i + 1 == len(cam_entries):
            progress_bar(i + 1, len(cam_entries), cam_name, start_time=t0)


def write_gt(writer, gt_entries):
    t0 = time.time()
    for i, (ts_ns, px, py, pz, qw, qx, qy, qz) in enumerate(gt_entries):
        msg = make_pose_stamped(ts_ns, px, py, pz, qw, qx, qy, qz)
        writer.write("/gt/pose", serialize_message(msg), ts_ns)
        if (i + 1) % 500 == 0 or i + 1 == len(gt_entries):
            progress_bar(i + 1, len(gt_entries), "/gt/pose", start_time=t0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Incremental rebuild: copy valid topics from old bag
# ═══════════════════════════════════════════════════════════════════════════════


def copy_good_topics(old_bag_path, writer, good_topics):
    if not good_topics:
        return

    print(f"  Copying {len(good_topics)} valid topic(s) from existing bag...")
    storage_options = rosbag2_py.StorageOptions(uri=old_bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    copied = 0
    total_read = 0
    t0 = time.time()
    while reader.has_next():
        topic, data, ts = reader.read_next()
        total_read += 1
        if topic in good_topics:
            writer.write(topic, data, ts)
            copied += 1
        if total_read % 500 == 0:
            progress_bar(copied, copied, "copying", start_time=t0)

    progress_bar(copied, copied, "copying", start_time=t0)
    del reader


# ═══════════════════════════════════════════════════════════════════════════════
#  Main conversion
# ═══════════════════════════════════════════════════════════════════════════════


def convert(dataset_path: str, output_path: str):
    """Convert UZH indoor dataset to ROS2 bag, reusing valid topics from existing bag."""

    left_txt = os.path.join(dataset_path, "left_images.txt")
    right_txt = os.path.join(dataset_path, "right_images.txt")
    imu_txt = os.path.join(dataset_path, "imu.txt")
    gt_txt = os.path.join(dataset_path, "groundtruth.txt")

    for p in [left_txt, right_txt, imu_txt]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found")
            sys.exit(1)

    has_gt = os.path.exists(gt_txt)

    # ── Read dataset files ──
    print("Reading dataset files...")
    cam0_entries = read_image_txt(left_txt)
    cam1_entries = read_image_txt(right_txt)
    imu_entries = read_imu_txt(imu_txt)
    gt_entries = read_groundtruth_txt(gt_txt) if has_gt else []
    print(f"  cam0: {len(cam0_entries)} frames")
    print(f"  cam1: {len(cam1_entries)} frames")
    print(f"  imu:  {len(imu_entries)} samples")
    print(f"  gt:   {len(gt_entries)} poses", "(note: GT only covers Vicon capture window)" if gt_entries else "")

    expected = compute_expected_counts(cam0_entries, cam1_entries, imu_entries, gt_entries)

    # ── Validate existing bag ──
    if os.path.exists(output_path):
        print(f"\nExisting bag found at {output_path} — validating...")
        is_valid, problems = validate_bag(output_path, expected)

        if is_valid:
            print("All topics valid. Nothing to do.")
            return

        print(f"\n{len(problems)} topic(s) need rebuilding:")
        for topic, (actual, exp) in sorted(problems.items()):
            status = "MISSING" if actual == 0 else f"WRONG COUNT ({actual} != {exp})"
            print(f"  {topic}: {status}")

        bad_topics = set(problems.keys())
        good_topics = set(expected.keys()) - bad_topics
        # camera_info and image topics are coupled: if either is bad, redo both
        for img_topic, info_topic in [("/cam0/image_raw", "/cam0/camera_info"),
                                       ("/cam1/image_raw", "/cam1/camera_info")]:
            if img_topic in bad_topics or info_topic in bad_topics:
                bad_topics.update([img_topic, info_topic])
                good_topics.discard(img_topic)
                good_topics.discard(info_topic)

        print(
            f"\nRebuilding bag — copying {len(good_topics)} good topic(s), "
            f"regenerating {len(bad_topics)}..."
        )

        old_path = output_path + ".old"
        if os.path.exists(old_path):
            shutil.rmtree(old_path)
        os.rename(output_path, old_path)
    else:
        print("\nNo existing bag — creating from scratch.")
        bad_topics = set(expected.keys())
        good_topics = set()
        old_path = None

    # ── Pre-compute fisheye undistortion maps ──
    print("Pre-computing fisheye undistortion maps...")
    cam0_map1, cam0_map2 = _build_undistort_maps(CAM0_INTRINSICS, CAM0_DISTORTION, RESOLUTION)
    cam1_map1, cam1_map2 = _build_undistort_maps(CAM1_INTRINSICS, CAM1_DISTORTION, RESOLUTION)

    # ── Create new bag ──
    storage_options = rosbag2_py.StorageOptions(uri=output_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    for topic_id, (topic_name, topic_type) in enumerate(ALL_TOPICS, start=1):
        topic_meta = rosbag2_py.TopicMetadata(
            id=topic_id, name=topic_name, type=topic_type, serialization_format="cdr"
        )
        writer.create_topic(topic_meta)

    # ── Copy valid messages from old bag ──
    if old_path and good_topics:
        copy_good_topics(old_path, writer, good_topics)

    # ── Write fresh data for bad topics ──
    if "/imu0" in bad_topics:
        write_imu(writer, imu_entries)

    if "/cam0/image_raw" in bad_topics:
        write_cam(writer, cam0_entries, dataset_path, "cam0",
                  CAM0_INTRINSICS, cam0_map1, cam0_map2)

    if "/cam1/image_raw" in bad_topics:
        write_cam(writer, cam1_entries, dataset_path, "cam1",
                  CAM1_INTRINSICS, cam1_map1, cam1_map2)

    if "/gt/pose" in bad_topics:
        if gt_entries:
            write_gt(writer, gt_entries)
        else:
            print("  Skipping /gt/pose — no groundtruth.txt found.")

    del writer

    # ── Clean up old bag ──
    if old_path and os.path.exists(old_path):
        shutil.rmtree(old_path)
        print("  Cleaned up old bag.")

    print(f"\nBag written to: {output_path}")
    print("Verify with: ros2 bag info " + output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert UZH FPV indoor dataset to ROS2 bag"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to UZH dataset directory (contains left_images.txt, imu.txt, etc.)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output bag path (directory, without .db3 extension)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force full rebuild even if bag exists"
    )
    args = parser.parse_args()

    if args.force and os.path.exists(args.output):
        print(f"--force: removing existing bag {args.output}")
        shutil.rmtree(args.output)

    convert(args.dataset, args.output)


if __name__ == "__main__":
    main()
