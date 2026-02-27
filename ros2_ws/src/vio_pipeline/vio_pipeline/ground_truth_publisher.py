import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # type: ignore
from geometry_msgs.msg import PoseStamped  # type: ignore
from nav_msgs.msg import Path, Odometry  # type: ignore

import numpy as np
import yaml


def stamp_to_ns(stamp):
    """Convert ROS2 stamp to nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _quat_to_rot(q):
    """Convert quaternion [x, y, z, w] to 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rot_to_quat(R):
    """Convert 3×3 rotation matrix to quaternion [x, y, z, w] (Shepperd method)."""
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
    return np.array([x, y, z, w])


def _pose_msg_to_matrix(pose):
    """Convert geometry_msgs/Pose to a 4×4 homogeneous transform matrix."""
    q = np.array(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )
    T = np.eye(4)
    T[:3, :3] = _quat_to_rot(q)
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


class GroundTruthPublisherNode(Node):
    """
    Re-publishes ground truth poses in the same world frame as PoseEstimationNode.

    Alignment
    ---------
    Our odometry world frame = cam0 optical frame at t=0.
    In Vicon coordinates that frame is:
        T_vicon_world = T_vicon_body_0 @ T_b_c0

    So the alignment transform is:
        T_align = inv(T_vicon_world)

    Applied per message:
        T_world_body_gt(t) = T_align @ T_vicon_body(t)

    At t=0 this evaluates to inv(T_b_c0), which is identical to what
    PoseEstimationNode publishes on the first frame — so both trajectories
    share the same starting pose and can be compared directly.
    """

    def __init__(self):
        super().__init__("ground_truth_publisher")

        self.declare_parameter("config_path", "")
        self.declare_parameter("gt_topic", "/gt/pose")

        config_path = self.get_parameter("config_path").value
        gt_topic = self.get_parameter("gt_topic").value

        self.load_config(config_path)
        self._setup_camera_params()

        # Set on the first GT message received
        self._T_align = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.gt_sub = self.create_subscription(
            PoseStamped, gt_topic, self.gt_callback, qos
        )

        self.pose_pub = self.create_publisher(PoseStamped, "/gt_pub/pose", 10)
        self.path_pub = self.create_publisher(Path, "/gt_pub/path", 10)
        self.odom_pub = self.create_publisher(Odometry, "/gt_pub/odometry", 10)

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        self.get_logger().info(
            f"GroundTruthPublisherNode initialized — listening on '{gt_topic}'"
        )

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config

    def _setup_camera_params(self):
        # Only T_b_c0 is needed: maps cam0 points → body frame (Kalibr T_BS convention)
        cam0 = self.config["cam0"]
        self.T_b_c0 = np.array(cam0["T_BS"], dtype=np.float64).reshape(4, 4)

    def gt_callback(self, msg):
        T_vicon_body = _pose_msg_to_matrix(msg.pose)

        if self._T_align is None:
            # World frame = initial body frame (same convention as odometry).
            # Strip only the initial body pose → both GT and odom start at identity.
            self._T_align = np.linalg.inv(T_vicon_body)
            self.get_logger().info("Ground truth alignment initialised")

        T_world_body = self._T_align @ T_vicon_body

        R = T_world_body[:3, :3]
        t = T_world_body[:3, 3]
        q = _rot_to_quat(R)
        stamp = msg.header.stamp

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(t[0])
        pose_msg.pose.position.y = float(t[1])
        pose_msg.pose.position.z = float(t[2])
        pose_msg.pose.orientation.x = float(q[0])
        pose_msg.pose.orientation.y = float(q[1])
        pose_msg.pose.orientation.z = float(q[2])
        pose_msg.pose.orientation.w = float(q[3])

        self.pose_pub.publish(pose_msg)

        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose_msg)
        self.path_pub.publish(self.path_msg)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose = pose_msg.pose
        self.odom_pub.publish(odom_msg)

        self.get_logger().debug(f"gt pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
