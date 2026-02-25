import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np


class GroundTruthPublisher(Node):
    """Republishes Vicon ground truth as Odometry and Path for RViz visualization."""

    def __init__(self):
        super().__init__("ground_truth_publisher")

        self.odom_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self.path_pub = self.create_publisher(Path, "/ground_truth/path", 10)

        self.create_subscription(
            TransformStamped,
            "/vicon/firefly_sbx/firefly_sbx",
            self.vicon_callback,
            100,
        )

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        # Store first pose as origin for calibration
        self.origin_T = None  # 4x4 transform of first Vicon reading

        self.get_logger().info("Ground Truth Publisher Started.")

    def vicon_callback(self, msg):
        t = msg.transform.translation
        r = msg.transform.rotation

        # Build 4x4 transform from Vicon reading
        T_cur = np.eye(4)
        T_cur[:3, :3] = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        T_cur[:3, 3] = [t.x, t.y, t.z]

        # Calibrate: store first pose and subtract from all
        if self.origin_T is None:
            self.origin_T = T_cur.copy()
            self.get_logger().info(
                f"Ground truth origin set: [{t.x:.2f}, {t.y:.2f}, {t.z:.2f}]"
            )

        # T_calibrated = T_origin_inv @ T_cur
        T_cal = np.linalg.inv(self.origin_T) @ T_cur
        pos = T_cal[:3, 3]
        quat = R.from_matrix(T_cal[:3, :3]).as_quat()

        # --- Publish as Odometry ---
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "ground_truth"

        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])

        odom.pose.pose.orientation.x = float(quat[0])
        odom.pose.pose.orientation.y = float(quat[1])
        odom.pose.pose.orientation.z = float(quat[2])
        odom.pose.pose.orientation.w = float(quat[3])

        self.odom_pub.publish(odom)

        # --- Append to Path ---
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose

        self.path_msg.header.stamp = msg.header.stamp
        self.path_msg.poses.append(pose)

        # Limit path length to avoid memory growth
        if len(self.path_msg.poses) > 5000:
            self.path_msg.poses = self.path_msg.poses[-5000:]

        self.path_pub.publish(self.path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
