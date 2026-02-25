import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
import os
from datetime import datetime


class TrajectoryComparator(Node):
    """
    Subscribes to VIO odometry and ground truth, time-aligns them,
    aligns coordinate frames, and writes a CSV with per-sample error diagnostics.

    Frame alignment: On the first matched pair, computes R_align from the
    orientation difference (VIO is in gravity-aligned world frame, GT is in
    initial body frame). All GT positions are rotated by R_align before
    computing position errors so the comparison is frame-consistent.
    """

    def __init__(self):
        super().__init__("trajectory_comparator")

        # Output CSV path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.expanduser(f"/mnt/d/GITHUB/VIO/vio_comparison_{timestamp}.csv")

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time_s",
            "gt_x", "gt_y", "gt_z",
            "vio_x", "vio_y", "vio_z",
            "err_x", "err_y", "err_z",
            "err_norm",
            "gt_roll", "gt_pitch", "gt_yaw",
            "vio_roll", "vio_pitch", "vio_yaw",
            "err_roll", "err_pitch", "err_yaw",
        ])

        # Buffers for time alignment
        self.gt_buffer = {}   # timestamp_ns -> (pos, quat)
        self.vio_buffer = {}  # timestamp_ns -> (pos, quat)

        self.match_tolerance_ns = 10_000_000  # 10ms tolerance for matching

        # Frame alignment rotation (computed from first matched pair)
        self.R_align = None

        # Subscribers
        self.create_subscription(
            Odometry, "/ground_truth/odom", self.gt_callback, 100
        )
        self.create_subscription(
            Odometry, "/odom/vio", self.vio_callback, 100
        )

        self.samples_written = 0
        self.start_time_ns = None

        self.get_logger().info(f"Trajectory Comparator writing to: {self.csv_path}")

    def _stamp_to_ns(self, stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def _extract_pose(self, msg):
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        quat = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])
        return pos, quat

    def gt_callback(self, msg):
        t_ns = self._stamp_to_ns(msg.header.stamp)
        pos, quat = self._extract_pose(msg)
        self.gt_buffer[t_ns] = (pos, quat)
        self._try_match(t_ns, is_gt=True)

    def vio_callback(self, msg):
        t_ns = self._stamp_to_ns(msg.header.stamp)
        pos, quat = self._extract_pose(msg)
        self.vio_buffer[t_ns] = (pos, quat)
        self._try_match(t_ns, is_gt=False)

    def _try_match(self, t_ns, is_gt):
        """Try to find a matching sample from the other source."""
        if is_gt:
            search_buffer = self.vio_buffer
            this_buffer = self.gt_buffer
        else:
            search_buffer = self.gt_buffer
            this_buffer = self.vio_buffer

        # Find closest timestamp in the other buffer
        best_key = None
        best_diff = float("inf")
        for key in list(search_buffer.keys()):
            diff = abs(key - t_ns)
            if diff < best_diff:
                best_diff = diff
                best_key = key

        if best_key is None or best_diff > self.match_tolerance_ns:
            return

        # Get matched pair
        if is_gt:
            gt_pos, gt_quat = this_buffer[t_ns]
            vio_pos, vio_quat = search_buffer[best_key]
        else:
            vio_pos, vio_quat = this_buffer[t_ns]
            gt_pos, gt_quat = search_buffer[best_key]

        # Compute frame alignment on first match
        if self.R_align is None:
            R_vio = R.from_quat(vio_quat)
            R_gt = R.from_quat(gt_quat)
            # R_align transforms GT frame -> VIO frame
            self.R_align = R_vio * R_gt.inv()
            self.get_logger().info(
                f"Frame alignment computed. "
                f"R_align euler: {self.R_align.as_euler('xyz', degrees=True)}"
            )

        # Compute time relative to start
        if self.start_time_ns is None:
            self.start_time_ns = min(t_ns, best_key)
        time_s = (t_ns - self.start_time_ns) / 1e9

        # Align GT position into VIO world frame before computing error
        gt_pos_aligned = self.R_align.apply(gt_pos)

        # Position error (both now in VIO world frame)
        err = vio_pos - gt_pos_aligned
        err_norm = np.linalg.norm(err)

        # Orientation: compute relative rotation error
        R_vio = R.from_quat(vio_quat)
        R_gt_aligned = self.R_align * R.from_quat(gt_quat)
        R_err = R_vio * R_gt_aligned.inv()
        err_euler = R_err.as_euler("xyz", degrees=True)

        # Euler angles for CSV (aligned GT and raw VIO)
        gt_euler = R_gt_aligned.as_euler("xyz", degrees=True)
        vio_euler = R_vio.as_euler("xyz", degrees=True)

        # Write CSV row
        self.csv_writer.writerow([
            f"{time_s:.4f}",
            f"{gt_pos_aligned[0]:.6f}", f"{gt_pos_aligned[1]:.6f}", f"{gt_pos_aligned[2]:.6f}",
            f"{vio_pos[0]:.6f}", f"{vio_pos[1]:.6f}", f"{vio_pos[2]:.6f}",
            f"{err[0]:.6f}", f"{err[1]:.6f}", f"{err[2]:.6f}",
            f"{err_norm:.6f}",
            f"{gt_euler[0]:.3f}", f"{gt_euler[1]:.3f}", f"{gt_euler[2]:.3f}",
            f"{vio_euler[0]:.3f}", f"{vio_euler[1]:.3f}", f"{vio_euler[2]:.3f}",
            f"{err_euler[0]:.3f}", f"{err_euler[1]:.3f}", f"{err_euler[2]:.3f}",
        ])
        self.csv_file.flush()
        self.samples_written += 1

        # Clean up matched entries from buffers to save memory
        if is_gt:
            self.gt_buffer.pop(t_ns, None)
            self.vio_buffer.pop(best_key, None)
        else:
            self.vio_buffer.pop(t_ns, None)
            self.gt_buffer.pop(best_key, None)

        # Prune old entries (keep buffer bounded)
        cutoff = t_ns - 500_000_000  # 0.5s
        for buf in [self.gt_buffer, self.vio_buffer]:
            stale = [k for k in buf if k < cutoff]
            for k in stale:
                del buf[k]

        if self.samples_written % 200 == 0:
            self.get_logger().info(
                f"Comparison: {self.samples_written} samples, "
                f"latest error: {err_norm:.4f}m"
            )

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(
            f"CSV written: {self.csv_path} ({self.samples_written} samples)"
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryComparator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
