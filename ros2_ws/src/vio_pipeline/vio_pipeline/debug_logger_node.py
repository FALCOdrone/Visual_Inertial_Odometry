#!/usr/bin/env python3
"""
debug_logger_node.py
====================
Subscribes to all VIO pipeline topics and writes line-buffered CSV logs for
offline analysis and plotting.

CSV files written to `output_dir` (default: <project_root>/tmp/):

  imu_raw.csv        Raw IMU from /imu0          (200 Hz)
  imu_processed.csv  Bias-corrected + LPF IMU    (200 Hz)
  pose_vio.csv       Visual odometry             (~20 Hz)
  pose_eskf.csv      ESKF fused pose             (200 Hz)
  pose_imu_dr.csv    IMU dead-reckoning          (200 Hz)
  pose_gt.csv        Ground truth                (variable)

All pose_*.csv files include pos_err_m and rot_err_deg columns, computed
by matching each estimate to the nearest ground-truth sample by timestamp.
These columns are left empty until the first GT message is received.

Parameters
----------
  output_dir   str    directory for CSV output  (default: <project_root>/tmp)
"""

import csv
import math
import os
import signal
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


# ── Minimal rotation helpers ───────────────────────────────────────────────────


def _quat_to_rot(q):
    """Quaternion [x, y, z, w] → 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _rot_err_deg(R_est, R_gt):
    """Geodesic rotation error between two rotation matrices, in degrees."""
    dR = R_gt.T @ R_est
    cos_t = float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cos_t))


# ── Node ──────────────────────────────────────────────────────────────────────


class DebugLoggerNode(Node):
    """
    Passive logger — subscribes to every pipeline topic and writes CSVs.
    Does not publish anything.
    """

    # How many GT samples to keep in the sliding look-up buffer
    _GT_BUF = 1000

    def __init__(self) -> None:
        super().__init__("debug_logger_node")

        _default_out = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tmp")
        )
        self.declare_parameter("output_dir", _default_out)
        out_dir = self.get_parameter("output_dir").value
        os.makedirs(out_dir, exist_ok=True)

        # ── Open CSV files ───────────────────────────────────────────────────
        self._fds:     dict = {}   # filename → file object
        self._writers: dict = {}   # filename → csv.DictWriter

        imu_fields = [
            "timestamp_ns",
            "gyro_x", "gyro_y", "gyro_z",
            "accel_x", "accel_y", "accel_z",
        ]
        pose_fields = [
            "timestamp_ns",
            "px", "py", "pz",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz",
            "pos_err_m", "rot_err_deg",
        ]
        gt_fields = [
            "timestamp_ns",
            "px", "py", "pz",
            "qx", "qy", "qz", "qw",
        ]

        for name, fields in [
            ("imu_raw.csv",        imu_fields),
            ("imu_processed.csv",  imu_fields),
            ("pose_vio.csv",       pose_fields),
            ("pose_eskf.csv",      pose_fields),
            ("pose_imu_dr.csv",    pose_fields),
            ("pose_gt.csv",        gt_fields),
        ]:
            path = os.path.join(out_dir, name)
            fd = open(path, "w", newline="", buffering=1)   # line-buffered
            w  = csv.DictWriter(fd, fieldnames=fields)
            w.writeheader()
            self._fds[name]     = fd
            self._writers[name] = w

        # ── Ground-truth sliding buffer for error computation ────────────────
        # Each entry: (timestamp_ns, px, py, pz, qx, qy, qz, qw)
        self._gt_buf: deque = deque(maxlen=self._GT_BUF)

        # ── QoS ─────────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Imu,      "/imu0",            self._cb_imu_raw,  qos_be)
        self.create_subscription(Imu,      "/imu/processed",   self._cb_imu_proc, qos_be)
        self.create_subscription(Odometry, "/imu/odometry",    self._cb_imu_dr,   qos_be)
        self.create_subscription(Odometry, "/vio/odometry",    self._cb_vio,      qos_be)
        self.create_subscription(Odometry, "/eskf/odometry",   self._cb_eskf,     qos_be)
        self.create_subscription(Odometry, "/gt_pub/odometry", self._cb_gt,       qos_be)

        signal.signal(signal.SIGTERM, self._on_sigterm)

        self.get_logger().info(
            f"DebugLoggerNode active — writing CSVs to '{out_dir}'"
        )

    # ── Signal handler ─────────────────────────────────────────────────────────

    def _on_sigterm(self, signum, frame) -> None:
        """Flush and close CSVs on SIGTERM (sent by ros2 launch on shutdown)."""
        self._flush_files()
        raise SystemExit(0)

    def _flush_files(self) -> None:
        for fd in self._fds.values():
            try:
                fd.flush()
                fd.close()
            except Exception:
                pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _ts(header) -> int:
        return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec

    def _gt_error(self, ts_ns: int, px, py, pz, qx, qy, qz, qw):
        """
        Look up the nearest GT sample and return (pos_err_m, rot_err_deg).
        Returns ("", "") when no GT data has arrived yet.
        """
        if not self._gt_buf:
            return "", ""
        _, gx, gy, gz, gqx, gqy, gqz, gqw = min(
            self._gt_buf, key=lambda r: abs(r[0] - ts_ns)
        )
        pos_err = math.sqrt((px - gx)**2 + (py - gy)**2 + (pz - gz)**2)
        rot_err = _rot_err_deg(
            _quat_to_rot([qx,  qy,  qz,  qw]),
            _quat_to_rot([gqx, gqy, gqz, gqw]),
        )
        return round(pos_err, 6), round(rot_err, 4)

    # ── IMU callbacks ──────────────────────────────────────────────────────────

    def _write_imu(self, writer, msg: Imu) -> None:
        writer.writerow({
            "timestamp_ns": self._ts(msg.header),
            "gyro_x":  msg.angular_velocity.x,
            "gyro_y":  msg.angular_velocity.y,
            "gyro_z":  msg.angular_velocity.z,
            "accel_x": msg.linear_acceleration.x,
            "accel_y": msg.linear_acceleration.y,
            "accel_z": msg.linear_acceleration.z,
        })

    def _cb_imu_raw(self,  msg: Imu) -> None:
        self._write_imu(self._writers["imu_raw.csv"], msg)

    def _cb_imu_proc(self, msg: Imu) -> None:
        self._write_imu(self._writers["imu_processed.csv"], msg)

    # ── Odometry callbacks ─────────────────────────────────────────────────────

    def _cb_gt(self, msg: Odometry) -> None:
        ts = self._ts(msg.header)
        p  = msg.pose.pose.position
        q  = msg.pose.pose.orientation
        self._gt_buf.append((ts, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        self._writers["pose_gt.csv"].writerow({
            "timestamp_ns": ts,
            "px": p.x, "py": p.y, "pz": p.z,
            "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w,
        })

    def _write_pose(self, writer, msg: Odometry) -> None:
        ts = self._ts(msg.header)
        p  = msg.pose.pose.position
        q  = msg.pose.pose.orientation
        v  = msg.twist.twist.linear
        pos_err, rot_err = self._gt_error(ts, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        writer.writerow({
            "timestamp_ns": ts,
            "px": p.x, "py": p.y, "pz": p.z,
            "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w,
            "vx": v.x, "vy": v.y, "vz": v.z,
            "pos_err_m":   pos_err,
            "rot_err_deg": rot_err,
        })

    def _cb_vio(self,    msg: Odometry) -> None:
        self._write_pose(self._writers["pose_vio.csv"],    msg)

    def _cb_eskf(self,   msg: Odometry) -> None:
        self._write_pose(self._writers["pose_eskf.csv"],   msg)

    def _cb_imu_dr(self, msg: Odometry) -> None:
        self._write_pose(self._writers["pose_imu_dr.csv"], msg)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self.get_logger().info("DebugLoggerNode: all CSV files closed.")
        self._flush_files()
        super().destroy_node()


# ── Entry point ────────────────────────────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = DebugLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
