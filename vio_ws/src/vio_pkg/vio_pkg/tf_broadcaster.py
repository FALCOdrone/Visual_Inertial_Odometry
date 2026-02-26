# =============================================================================
# VIO TF Broadcaster – Publishes the full TF frame tree for RViz and tf2 tools
#
# Dynamic transforms  (from odometry topics):
#   odom → base_link      (EKF-fused pose)
#   odom → visual_odom    (frontend VO-only pose)
#   odom → ground_truth   (Vicon ground truth pose)
#
# Static transforms  (sensor mounting):
#   base_link → imu_link     (identity – IMU co-located with body origin)
#   base_link → camera_link  (R_body_cam rotation from EuRoC calibration)
# =============================================================================

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np


class VIOTFBroadcaster(Node):
    """Centralised TF broadcaster for the VIO pipeline."""

    def __init__(self):
        super().__init__("vio_tf_broadcaster")

        # --- Dynamic TF broadcaster ---
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Static TF broadcaster ---
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        # --- Subscribe to odometry topics ---
        self.create_subscription(
            Odometry, "/odom/vio_ekf", self.ekf_callback, 10
        )
        self.create_subscription(
            PoseWithCovarianceStamped, "/vio/visual_odom", self.vo_callback, 10
        )
        self.create_subscription(
            Odometry, "/ground_truth/odom", self.gt_callback, 10
        )

        self.get_logger().info("VIO TF Broadcaster started.")

    # ======================== Static Transforms ========================
    def _publish_static_transforms(self):
        """Publish sensor-mounting transforms (constant, sent once)."""
        static_transforms = []

        # --- base_link → imu_link (identity) ---
        t_imu = TransformStamped()
        t_imu.header.stamp = self.get_clock().now().to_msg()
        t_imu.header.frame_id = "base_link"
        t_imu.child_frame_id = "imu_link"
        t_imu.transform.rotation.w = 1.0  # identity quaternion
        static_transforms.append(t_imu)

        # --- base_link → camera_link (R_body_cam from EuRoC calibration) ---
        # Camera observes: X-right, Y-down, Z-forward
        # Body frame (FLU): X-forward, Y-left, Z-up
        # Mapping body ← cam:  body_X = cam_Z,  body_Y = -cam_X,  body_Z = -cam_Y
        # So cam ← body (inverse) is the static TF from base_link to camera_link
        R_body_cam = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ], dtype=np.float64)
        # TF convention: rotation maps child (camera) → parent (body), = R_body_cam
        q = R.from_matrix(R_body_cam).as_quat()  # x, y, z, w

        t_cam = TransformStamped()
        t_cam.header.stamp = self.get_clock().now().to_msg()
        t_cam.header.frame_id = "base_link"
        t_cam.child_frame_id = "camera_link"
        t_cam.transform.rotation.x = float(q[0])
        t_cam.transform.rotation.y = float(q[1])
        t_cam.transform.rotation.z = float(q[2])
        t_cam.transform.rotation.w = float(q[3])
        static_transforms.append(t_cam)

        self.static_tf_broadcaster.sendTransform(static_transforms)
        self.get_logger().info(
            "Published static TFs: base_link→imu_link, base_link→camera_link"
        )

    # ====================== Dynamic Transforms ======================
    def _odom_to_tf(self, msg, child_frame_id):
        """Convert an Odometry message to a TF and broadcast it."""
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id  # "odom"
        t.child_frame_id = child_frame_id

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation.x = msg.pose.pose.orientation.x
        t.transform.rotation.y = msg.pose.pose.orientation.y
        t.transform.rotation.z = msg.pose.pose.orientation.z
        t.transform.rotation.w = msg.pose.pose.orientation.w

        self.tf_broadcaster.sendTransform(t)

    def ekf_callback(self, msg):
        """odom → base_link (EKF-fused pose)."""
        self._odom_to_tf(msg, "base_link")

    def vo_callback(self, msg):
        """odom → visual_odom (frontend VO-only pose)."""
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id  # "odom"
        t.child_frame_id = "visual_odom"

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation.x = msg.pose.pose.orientation.x
        t.transform.rotation.y = msg.pose.pose.orientation.y
        t.transform.rotation.z = msg.pose.pose.orientation.z
        t.transform.rotation.w = msg.pose.pose.orientation.w

        self.tf_broadcaster.sendTransform(t)

    def gt_callback(self, msg):
        """odom → ground_truth (Vicon ground truth pose)."""
        self._odom_to_tf(msg, "ground_truth")


def main(args=None):
    rclpy.init(args=args)
    node = VIOTFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
