#!/usr/bin/env python3
"""
tf_publisher_node.py
====================
Publishes the conventional ROS 2 TF tree for the VIO pipeline.

Static transforms  (sensor extrinsics, loaded from euroc_params.yaml)
----------------------------------------------------------------------
  base_link → imu0    identity  (EuRoC: IMU frame == body frame)
  base_link → cam0    T_BS from cam0 config  (sensor pose in body frame)
  base_link → cam1    T_BS from cam1 config

Dynamic transform  (re-broadcast from ESKF fused odometry, ~200 Hz)
---------------------------------------------------------------------
  map → base_link     from /eskf/odometry

T_BS convention (EuRoC)
------------------------
T_BS is the 4×4 homogeneous transform that maps a point in the sensor (S)
frame into the body (B) frame:  p_B = T_BS · p_S.
In TF2 terms the transform labelled  parent=base_link / child=cam0
converts cam0 coords → base_link coords, so T_BS is used directly.

Subscriptions
-------------
  /eskf/odometry   nav_msgs/Odometry   fused pose @ ~200 Hz

Publications
------------
  /tf_static   geometry_msgs/TransformStamped  (StaticTransformBroadcaster)
  /tf          geometry_msgs/TransformStamped  (TransformBroadcaster)

Parameters
----------
  config_path   str   absolute path to euroc_params.yaml (or equivalent)
"""

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

import tf2_ros


# ── Helpers ────────────────────────────────────────────────────────────────────


def _rot_to_quat(R: np.ndarray):
    """3×3 rotation matrix → (x, y, z, w) quaternion (Shepperd method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _mat4_to_stamped(
    T: np.ndarray, parent: str, child: str, stamp
) -> TransformStamped:
    """4×4 homogeneous transform → TransformStamped (parent → child)."""
    R = T[:3, :3]
    t = T[:3, 3]
    x, y, z, w = _rot_to_quat(R)

    ts = TransformStamped()
    ts.header.stamp = stamp
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = float(t[0])
    ts.transform.translation.y = float(t[1])
    ts.transform.translation.z = float(t[2])
    ts.transform.rotation.x = x
    ts.transform.rotation.y = y
    ts.transform.rotation.z = z
    ts.transform.rotation.w = w
    return ts


# ── Node ───────────────────────────────────────────────────────────────────────


class TfPublisherNode(Node):
    """
    Publishes sensor extrinsic static TFs and re-broadcasts the ESKF
    odometry as a dynamic map → base_link transform.
    """

    def __init__(self) -> None:
        super().__init__("tf_publisher_node")

        self.declare_parameter("config_path", "")
        config_path = self.get_parameter("config_path").value

        # ── TF broadcasters ──────────────────────────────────────────────────
        self._static_br = tf2_ros.StaticTransformBroadcaster(self)
        self._br = tf2_ros.TransformBroadcaster(self)

        # ── Publish static sensor extrinsics ─────────────────────────────────
        self._publish_static_transforms(config_path)

        # ── Subscribe to ESKF odometry ───────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.create_subscription(Odometry, "/eskf/odometry", self._odom_cb, qos_be)

        self.get_logger().info(
            "TfPublisherNode ready — publishing map→base_link from /eskf/odometry."
        )

    # ── Static extrinsics ──────────────────────────────────────────────────────

    def _publish_static_transforms(self, config_path: str) -> None:
        now = self.get_clock().now().to_msg()
        transforms = []
        T_imu = np.eye(4)

        if config_path:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)

                def _load_T(section: str) -> np.ndarray:
                    flat = cfg[section]["T_BS"]
                    return np.array(flat, dtype=np.float64).reshape(4, 4)

                T_imu = _load_T("imu")
                T_cam0 = _load_T("cam0")
                T_cam1 = _load_T("cam1")

                transforms.append(_mat4_to_stamped(T_cam0, "base_link", "cam0", now))
                transforms.append(_mat4_to_stamped(T_cam1, "base_link", "cam1", now))
                self.get_logger().info(
                    f"Loaded sensor extrinsics from '{config_path}'."
                )
            except Exception as exc:
                self.get_logger().warn(
                    f"Could not load config '{config_path}': {exc}. "
                    "Falling back to identity transforms for cameras."
                )
                transforms.append(
                    _mat4_to_stamped(np.eye(4), "base_link", "cam0", now)
                )
                transforms.append(
                    _mat4_to_stamped(np.eye(4), "base_link", "cam1", now)
                )
        else:
            self.get_logger().warn(
                "No config_path provided — publishing identity static transforms."
            )
            transforms.append(_mat4_to_stamped(np.eye(4), "base_link", "cam0", now))
            transforms.append(_mat4_to_stamped(np.eye(4), "base_link", "cam1", now))

        transforms.append(_mat4_to_stamped(T_imu, "base_link", "imu0", now))
        self._static_br.sendTransform(transforms)

    # ── Dynamic map → base_link ────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        """Re-publish ESKF odometry as map → base_link TF."""
        ts = TransformStamped()
        ts.header = msg.header          # frame_id == "map"
        ts.child_frame_id = "base_link"
        ts.transform.translation.x = msg.pose.pose.position.x
        ts.transform.translation.y = msg.pose.pose.position.y
        ts.transform.translation.z = msg.pose.pose.position.z
        ts.transform.rotation = msg.pose.pose.orientation
        self._br.sendTransform(ts)


# ── Entry point ────────────────────────────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = TfPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
