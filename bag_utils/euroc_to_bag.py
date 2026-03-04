#!/usr/bin/env python3
"""
Convert EuRoC MAV dataset to a ROS2 bag file (.db3).

If the bag already exists, validates per-topic message counts and only
rebuilds topics that are missing or have incorrect counts.

Usage:
    python3 euroc_to_bag.py --dataset dataset/mav0 --output bags/euroc_mav0
"""

import argparse
import csv
import os
import shutil
import struct
import sys
import time

import cv2
import numpy as np

from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from builtin_interfaces.msg import Time
import rosbag2_py


# ── EuRoC camera calibration (hardcoded for MH_01_easy / V1_01_easy etc.) ─────
CAM0_INTRINSICS = [458.654, 457.296, 367.215, 248.375]
CAM0_DISTORTION = [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]

CAM1_INTRINSICS = [457.587, 456.134, 379.999, 255.238]
CAM1_DISTORTION = [-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05]

RESOLUTION = (752, 480)  # width, height


# ── Topic definitions ──────────────────────────────────────────────────────────
# Each topic: (name, ros_type)
ALL_TOPICS = [
    ("/cam0/image_raw", "sensor_msgs/msg/Image"),
    ("/cam1/image_raw", "sensor_msgs/msg/Image"),
    ("/cam0/camera_info", "sensor_msgs/msg/CameraInfo"),
    ("/cam1/camera_info", "sensor_msgs/msg/CameraInfo"),
    ("/imu0", "sensor_msgs/msg/Imu"),
    ("/gt/pose", "geometry_msgs/msg/PoseStamped"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Progress bar
# ═══════════════════════════════════════════════════════════════════════════════


def progress_bar(current, total, label="", bar_width=40, start_time=None):
    """Print an in-place progress bar to stderr."""
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
#  Message builders
# ═══════════════════════════════════════════════════════════════════════════════


def ns_to_time_msg(timestamp_ns: int) -> Time:
    t = Time()
    t.sec = int(timestamp_ns // 1_000_000_000)
    t.nanosec = int(timestamp_ns % 1_000_000_000)
    return t


def make_camera_info(intrinsics, distortion, resolution, timestamp_ns, frame_id):
    ci = CameraInfo()
    ci.header.stamp = ns_to_time_msg(timestamp_ns)
    ci.header.frame_id = frame_id
    ci.width = resolution[0]
    ci.height = resolution[1]
    ci.distortion_model = "plumb_bob"
    ci.d = list(distortion) + [0.0] if len(distortion) == 4 else list(distortion)
    fx, fy, cx, cy = intrinsics
    ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return ci


def make_image_msg(image_path, timestamp_ns, frame_id):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    msg = Image()
    msg.header.stamp = ns_to_time_msg(timestamp_ns)
    msg.header.frame_id = frame_id
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.step = img.shape[1]
    msg.data = img.tobytes()
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
#  CSV readers
# ═══════════════════════════════════════════════════════════════════════════════


def read_cam_csv(csv_path):
    entries = []
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            if row[0].startswith("#"):
                continue
            entries.append((int(row[0]), row[1].strip()))
    return entries


def read_imu_csv(csv_path):
    entries = []
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            if row[0].startswith("#"):
                continue
            entries.append(
                (
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                )
            )
    return entries


def read_groundtruth_csv(csv_path):
    entries = []
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            if row[0].startswith("#"):
                continue
            entries.append(
                (
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                )
            )
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
#  Bag validation
# ═══════════════════════════════════════════════════════════════════════════════


def read_bag_topic_counts(bag_path):
    """
    Open an existing bag and return {topic_name: message_count}.
    Returns None if the bag cannot be opened.
    """
    try:
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_counts = {}
        for topic_meta in reader.get_all_topics_and_types():
            topic_counts[topic_meta.name] = 0

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
    """
    Return {topic_name: expected_message_count} for every topic.
    cam_info topics mirror their image topics 1:1.
    """
    return {
        "/cam0/image_raw": len(cam0_entries),
        "/cam1/image_raw": len(cam1_entries),
        "/cam0/camera_info": len(cam0_entries),
        "/cam1/camera_info": len(cam1_entries),
        "/imu0": len(imu_entries),
        "/gt/pose": len(gt_entries),
    }


def validate_bag(bag_path, expected):
    """
    Compare existing bag against expected counts.
    Returns (is_valid, {topic: (bag_count, expected_count)} for bad topics).
    """
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
#  Bag writing helpers (per topic group)
# ═══════════════════════════════════════════════════════════════════════════════


def write_imu(writer, imu_entries):
    t0 = time.time()
    for i, (ts_ns, gx, gy, gz, ax, ay, az) in enumerate(imu_entries):
        msg = make_imu_msg(ts_ns, gx, gy, gz, ax, ay, az)
        writer.write("/imu0", serialize_message(msg), ts_ns)
        if (i + 1) % 500 == 0 or i + 1 == len(imu_entries):
            progress_bar(i + 1, len(imu_entries), "/imu0", start_time=t0)


def write_cam(writer, cam_entries, data_dir, cam_name, intrinsics, distortion):
    img_topic = f"/{cam_name}/image_raw"
    info_topic = f"/{cam_name}/camera_info"
    t0 = time.time()
    for i, (ts_ns, filename) in enumerate(cam_entries):
        img_path = os.path.join(data_dir, filename)
        img_msg = make_image_msg(img_path, ts_ns, cam_name)
        if img_msg is None:
            sys.stderr.write(f"\n    WARNING: Could not read {img_path}\n")
            continue
        writer.write(img_topic, serialize_message(img_msg), ts_ns)
        ci_msg = make_camera_info(intrinsics, distortion, RESOLUTION, ts_ns, cam_name)
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
#  Core logic: copy good messages from old bag, write fresh for bad topics
# ═══════════════════════════════════════════════════════════════════════════════


def copy_good_topics(old_bag_path, writer, good_topics):
    """
    Read the old bag and re-write only messages belonging to good_topics.
    """
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


def convert(dataset_path: str, output_path: str):
    """Convert EuRoC dataset to ROS2 bag, reusing valid topics from existing bag."""

    cam0_csv = os.path.join(dataset_path, "cam0", "data.csv")
    cam1_csv = os.path.join(dataset_path, "cam1", "data.csv")
    imu_csv = os.path.join(dataset_path, "imu0", "data.csv")
    gt_csv = os.path.join(dataset_path, "state_groundtruth_estimate0", "data.csv")
    cam0_data_dir = os.path.join(dataset_path, "cam0", "data")
    cam1_data_dir = os.path.join(dataset_path, "cam1", "data")

    for p in [cam0_csv, cam1_csv, imu_csv, cam0_data_dir, cam1_data_dir]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found")
            sys.exit(1)

    has_gt = os.path.exists(gt_csv)

    # ── Read dataset CSVs ──
    print("Reading CSV files...")
    cam0_entries = read_cam_csv(cam0_csv)
    cam1_entries = read_cam_csv(cam1_csv)
    imu_entries = read_imu_csv(imu_csv)
    gt_entries = read_groundtruth_csv(gt_csv) if has_gt else []
    print(f"  cam0: {len(cam0_entries)} frames")
    print(f"  cam1: {len(cam1_entries)} frames")
    print(f"  imu:  {len(imu_entries)} samples")
    print(f"  gt:   {len(gt_entries)} poses")

    expected = compute_expected_counts(
        cam0_entries, cam1_entries, imu_entries, gt_entries
    )

    # ── Validate existing bag ──
    if os.path.exists(output_path):
        print(f"\nExisting bag found at {output_path} — validating...")
        is_valid, problems = validate_bag(output_path, expected)

        if is_valid:
            print("All topics valid. Nothing to do.")
            return

        # Report problems
        print(f"\n{len(problems)} topic(s) need rebuilding:")
        for topic, (actual, exp) in sorted(problems.items()):
            status = "MISSING" if actual == 0 else f"WRONG COUNT ({actual} != {exp})"
            print(f"  {topic}: {status}")

        bad_topics = set(problems.keys())
        good_topics = set(expected.keys()) - bad_topics
        # camera_info and image topics are coupled: if either is bad, redo both
        cam_pairs = [
            ("/cam0/image_raw", "/cam0/camera_info"),
            ("/cam1/image_raw", "/cam1/camera_info"),
        ]
        for img_topic, info_topic in cam_pairs:
            if img_topic in bad_topics or info_topic in bad_topics:
                bad_topics.add(img_topic)
                bad_topics.add(info_topic)
                good_topics.discard(img_topic)
                good_topics.discard(info_topic)

        print(
            f"\nRebuilding bag — copying {len(good_topics)} good topic(s), regenerating {len(bad_topics)}..."
        )

        # Move old bag aside for reading
        old_path = output_path + ".old"
        if os.path.exists(old_path):
            shutil.rmtree(old_path)
        os.rename(output_path, old_path)
    else:
        print("\nNo existing bag — creating from scratch.")
        bad_topics = set(expected.keys())
        good_topics = set()
        old_path = None

    # ── Create new bag ──
    storage_options = rosbag2_py.StorageOptions(uri=output_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    # Register ALL topics (even ones we'll copy — writer needs them declared)
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
        write_cam(
            writer,
            cam0_entries,
            cam0_data_dir,
            "cam0",
            CAM0_INTRINSICS,
            CAM0_DISTORTION,
        )

    if "/cam1/image_raw" in bad_topics:
        write_cam(
            writer,
            cam1_entries,
            cam1_data_dir,
            "cam1",
            CAM1_INTRINSICS,
            CAM1_DISTORTION,
        )

    if "/gt/pose" in bad_topics:
        if gt_entries:
            write_gt(writer, gt_entries)
        else:
            print("  Skipping /gt/pose — no ground truth CSV found.")

    del writer

    # ── Clean up old bag ──
    if old_path and os.path.exists(old_path):
        shutil.rmtree(old_path)
        print("  Cleaned up old bag.")

    print(f"\nBag written to: {output_path}")
    print("Verify with: ros2 bag info " + output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert EuRoC MAV dataset to ROS2 bag"
    )
    parser.add_argument("--dataset", required=True, help="Path to EuRoC mav0 directory")
    parser.add_argument(
        "--output", required=True, help="Output bag path (without .db3 extension)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force full rebuild even if bag exists"
    )
    args = parser.parse_args()

    if args.force and os.path.exists(args.output):
        print(f"--force: removing existing bag {args.output}")
        shutil.rmtree(args.output)

    convert(args.dataset, args.output)


if __name__ == "__main__":
    main()
