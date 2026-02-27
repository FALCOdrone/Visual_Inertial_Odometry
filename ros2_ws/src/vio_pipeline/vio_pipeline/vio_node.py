import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from sensor_msgs.msg import Image  # type: ignore
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # type: ignore
from message_filters import Subscriber, ApproximateTimeSynchronizer  # type: ignore
from geometry_msgs.msg import PoseStamped  # type: ignore
from nav_msgs.msg import Path, Odometry  # type: ignore

import numpy as np
import cv2
import yaml

from vio_pipeline.feature_extraction import FeatureExtractor


def stamp_to_ns(stamp):
    """Convert ROS2 stamp to nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


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


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")

        self.declare_parameter("config_path", "")
        self.declare_parameter("device", "")
        self.declare_parameter("min_tracks", 10)
        self.declare_parameter("circular_check_threshold", 2.0)
        self.declare_parameter("max_depth", 30.0)          # metres — discard far points
        self.declare_parameter("min_inlier_ratio", 0.4)    # RANSAC inliers / candidates
        self.declare_parameter("max_translation", 0.5)     # metres per frame
        self.declare_parameter("max_rotation_deg", 30.0)   # degrees per frame

        config_path = self.get_parameter("config_path").value
        device_param = self.get_parameter("device").value
        self.min_tracks = self.get_parameter("min_tracks").value
        self.circular_threshold = self.get_parameter("circular_check_threshold").value
        self.max_depth = self.get_parameter("max_depth").value
        self.min_inlier_ratio = self.get_parameter("min_inlier_ratio").value
        self.max_translation = self.get_parameter("max_translation").value
        self.max_rotation_deg = self.get_parameter("max_rotation_deg").value

        self.load_config(config_path)
        self._setup_camera_params()

        device_kwarg = {"device": device_param} if device_param else {}
        self.extractor = FeatureExtractor(**device_kwarg)

        if self.extractor.superpoint is None:
            self.get_logger().fatal(
                "SuperPoint/LightGlue models failed to load. "
                "Install with: pip install lightglue"
            )
            raise RuntimeError("lightglue not installed")

        # Feature tracking state
        self._prev_left_features = None
        self._prev_right_features = None

        # World frame = initial body frame (FLU: x-fwd, y-left, z-up at t=0).
        # Initialising with T_b_c0 ensures T_world_body_0 = T_b_c0 @ inv(T_b_c0) = I,
        # so the body starts at the origin with identity orientation.
        self.T_world_cam0 = self.T_b_c0.copy()

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        self._setup_ros_topics()
        self.get_logger().info("PoseEstimationNode initialized")

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config
        self.get_logger().info("Config loaded successfully")

    def _setup_camera_params(self):
        cam0 = self.config["cam0"]
        cam1 = self.config["cam1"]

        fx0, fy0, cx0, cy0 = cam0["intrinsics"]
        fx1, fy1, cx1, cy1 = cam1["intrinsics"]

        self.cam0_K = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]])
        self.cam1_K = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])

        # Distortion coefficients (used to undistort keypoints before geometry)
        self.cam0_dist = np.array(cam0["distortion"], dtype=np.float64)
        self.cam1_dist = np.array(cam1["distortion"], dtype=np.float64)

        # Kalibr T_BS convention: p_body = T_b_c @ p_cam
        self.T_b_c0 = np.array(cam0["T_BS"], dtype=np.float64).reshape(4, 4)
        self.T_b_c1 = np.array(cam1["T_BS"], dtype=np.float64).reshape(4, 4)

        # Transform that maps cam0 points → cam1 frame
        self.T_c1_c0 = np.linalg.inv(self.T_b_c1) @ self.T_b_c0

        # Stereo projection matrices (cam0 as reference frame)
        R = self.T_c1_c0[:3, :3]
        t = self.T_c1_c0[:3, 3:]
        self.P0 = self.cam0_K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P1 = self.cam1_K @ np.hstack([R, t])

    def _setup_ros_topics(self):
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cam0_sub = Subscriber(self, Image, "/cam0/image_raw", qos_profile=qos)
        self.cam1_sub = Subscriber(self, Image, "/cam1/image_raw", qos_profile=qos)

        self.time_sync = ApproximateTimeSynchronizer(
            [self.cam0_sub, self.cam1_sub], queue_size=10, slop=0.1
        )
        self.time_sync.registerCallback(self.stereo_callback)

        self.pose_pub = self.create_publisher(PoseStamped, "/vio/pose", 10)
        self.path_pub = self.create_publisher(Path, "/vio/path", 10)
        self.odom_pub = self.create_publisher(Odometry, "/vio/odometry", 10)

        self.get_logger().info("ROS topics initialized")

    def _image_msg_to_numpy(self, msg):
        """Convert sensor_msgs/Image to a grayscale numpy array."""
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding in ("mono8", "8UC1"):
            return img[:, :, 0]
        code = cv2.COLOR_RGB2GRAY if msg.encoding == "rgb8" else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)

    def stereo_callback(self, cam0_msg, cam1_msg):
        """Process a synchronized stereo pair: extract features, estimate pose."""
        ts_ns = stamp_to_ns(cam0_msg.header.stamp)
        stamp = cam0_msg.header.stamp

        left_img = self._image_msg_to_numpy(cam0_msg)
        right_img = self._image_msg_to_numpy(cam1_msg)

        result = self.extractor.process_full_frame(
            left_img,
            right_img,
            prev_left_features=self._prev_left_features,
            prev_right_features=self._prev_right_features,
            pixel_threshold=self.circular_threshold,
        )

        if result is None:
            self.get_logger().warn("Feature extraction returned no result")
            return

        # Always update stored features so the next frame has a prior
        self._prev_left_features = result["left_features"]
        self._prev_right_features = result["right_features"]

        tracks = result["circular_tracks"]
        if tracks is None:
            self.get_logger().info(f"ts={ts_ns} | first frame — no pose update")
            return

        if tracks["count"] < self.min_tracks:
            self.get_logger().warn(
                f"ts={ts_ns} | only {tracks['count']} tracks (min={self.min_tracks}), skipping"
            )
            return

        # Undistort all keypoints once so triangulation and PnP use a consistent model
        kpts_l_prev_u = self._undistort_points(tracks["kpts_l_prev"], self.cam0_K, self.cam0_dist)
        kpts_r_prev_u = self._undistort_points(tracks["kpts_r_prev"], self.cam1_K, self.cam1_dist)
        kpts_l_curr_u = self._undistort_points(tracks["kpts_l_curr"], self.cam0_K, self.cam0_dist)

        # --- Triangulate 3D landmarks from the previous stereo pair ---
        pts3d = self._triangulate(kpts_l_prev_u, kpts_r_prev_u)

        # --- PnP: find cam0_curr pose relative to cam0_prev ---
        T_rel = self._solve_pnp(pts3d, kpts_l_curr_u, ts_ns)
        if T_rel is None:
            return

        # T_rel: p_curr = T_rel @ p_prev  →  cam0_curr in world frame:
        self.T_world_cam0 = self.T_world_cam0 @ np.linalg.inv(T_rel)

        self._publish_pose(stamp, ts_ns)

    def _undistort_points(self, pts, K, dist):
        """Undistort 2D keypoints back into ideal pixel coordinates."""
        if pts.shape[0] == 0:
            return pts
        undist = cv2.undistortPoints(
            pts.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K
        )
        return undist.reshape(-1, 2)

    def _triangulate(self, kpts_l, kpts_r):
        """Triangulate stereo correspondences. Returns (N, 3) points in cam0 frame."""
        pts4d = cv2.triangulatePoints(
            self.P0,
            self.P1,
            kpts_l.T.astype(np.float64),
            kpts_r.T.astype(np.float64),
        )
        return (pts4d[:3] / pts4d[3]).T  # (N, 3)

    def _solve_pnp(self, pts3d, kpts2d, ts_ns):
        """
        Estimate relative pose via PnP with layered outlier rejection.

        pts3d:  (N, 3) landmarks in cam0_prev frame
        kpts2d: (N, 2) observations in cam0_curr image (undistorted)

        Returns 4×4 T such that p_curr = T @ p_prev, or None on failure.
        """
        # Depth filter: keep only points in a plausible stereo range
        valid = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < self.max_depth)
        pts3d_v = pts3d[valid]
        pts2d_v = kpts2d[valid].astype(np.float64)

        if len(pts3d_v) < 6:
            self.get_logger().warn(
                f"ts={ts_ns} | too few depth-valid points ({len(pts3d_v)})"
            )
            return None

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_v,
            pts2d_v,
            self.cam0_K,
            None,  # keypoints already undistorted
            iterationsCount=200,
            reprojectionError=2.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )

        n_in = len(inliers) if inliers is not None else 0
        if not ok or inliers is None or n_in < 6:
            self.get_logger().warn(f"ts={ts_ns} | PnP RANSAC failed (inliers={n_in})")
            return None

        # Inlier ratio guard: low ratio → RANSAC found a small bad consensus
        inlier_ratio = n_in / len(pts3d_v)
        if inlier_ratio < self.min_inlier_ratio:
            self.get_logger().warn(
                f"ts={ts_ns} | inlier ratio too low ({inlier_ratio:.2f} < {self.min_inlier_ratio})"
            )
            return None

        # Refine on inliers with iterative non-linear optimisation
        inlier_idx = inliers.ravel()
        _, rvec, tvec = cv2.solvePnP(
            pts3d_v[inlier_idx],
            pts2d_v[inlier_idx],
            self.cam0_K,
            None,
            rvec, tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        # Motion sanity check: reject physically impossible per-frame motion
        translation = np.linalg.norm(t)
        angle_deg = np.degrees(
            np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        )
        if translation > self.max_translation:
            self.get_logger().warn(
                f"ts={ts_ns} | translation too large ({translation:.3f}m > {self.max_translation}m), skipping"
            )
            return None
        if angle_deg > self.max_rotation_deg:
            self.get_logger().warn(
                f"ts={ts_ns} | rotation too large ({angle_deg:.1f}° > {self.max_rotation_deg}°), skipping"
            )
            return None

        self.get_logger().info(
            f"ts={ts_ns} | PnP inliers={n_in}/{len(pts3d_v)} ({inlier_ratio:.0%}) "
            f"t={translation:.3f}m rot={angle_deg:.1f}°"
        )

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def _publish_pose(self, stamp, ts_ns):
        """Publish current pose as PoseStamped, Path, and Odometry.

        Transforms cam0 optical pose → body frame so that child_frame_id='base_link'
        is correct for EKF fusion (robot_localization or similar).
        """
        # Express body (base_link) in world frame
        T_world_body = self.T_world_cam0 @ np.linalg.inv(self.T_b_c0)
        R = T_world_body[:3, :3]
        t = T_world_body[:3, 3]
        q = _rot_to_quat(R)

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

        self.get_logger().info(f"ts={ts_ns} | pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PoseEstimationNode()
    except RuntimeError as e:
        print(f"[FATAL] Failed to start PoseEstimationNode: {e}")
        rclpy.shutdown()
        return
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
