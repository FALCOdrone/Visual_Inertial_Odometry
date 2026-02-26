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
        self.origin_pos = None  # position of first Vicon reading
        self.origin_rot_inv = None  # inverse rotation of first Vicon reading

        self.get_logger().info("Ground Truth Publisher Started.")

    def vicon_callback(self, msg):
        t = msg.transform.translation
        r = msg.transform.rotation

        # Store first pose (position + orientation) as origin
        if self.origin_pos is None:
            self.origin_pos = np.array([t.x, t.y, t.z])
            self.origin_rot_inv = R.from_quat(
                [r.x, r.y, r.z, r.w]
            ).inv()
            self.get_logger().info(
                f"Ground truth origin set: [{t.x:.2f}, {t.y:.2f}, {t.z:.2f}]"
            )

        # Express position in the initial body frame so it matches the EKF frame
        pos_world = np.array([t.x, t.y, t.z]) - self.origin_pos
        pos = self.origin_rot_inv.apply(pos_world)

        # Relative orientation w.r.t. initial pose
        quat_current = R.from_quat([r.x, r.y, r.z, r.w])
        q_rel = (self.origin_rot_inv * quat_current).as_quat()  # x,y,z,w

        # --- Publish as Odometry ---
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "ground_truth"

        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])

        odom.pose.pose.orientation.x = float(q_rel[0])
        odom.pose.pose.orientation.y = float(q_rel[1])
        odom.pose.pose.orientation.z = float(q_rel[2])
        odom.pose.pose.orientation.w = float(q_rel[3])

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
