import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from sensor_msgs.msg import Image  # type: ignore
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # type: ignore
from message_filters import Subscriber, ApproximateTimeSynchronizer  # type: ignore
from geometry_msgs.msg import PoseStamped, Vector3Stamped  # type: ignore
from nav_msgs.msg import Path, Odometry  # type: ignore

import numpy as np
import cv2
import yaml

from vio_pipeline.feature_tracking_KLT import FeatureExtractor


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


def _rot_to_rpy(R):
    """Rotation matrix → (roll, pitch, yaw) in degrees (ZYX / intrinsic XYZ)."""
    pitch = np.arcsin(-R[2, 0])
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")
        cv2.setRNGSeed(0)  # make solvePnPRansac deterministic across runs

        self.declare_parameter("config_path", "")
        # VIO pose estimation params
        self.declare_parameter("min_tracks", 10)
        self.declare_parameter("circular_check_threshold", 2.0)
        self.declare_parameter("max_depth", 30.0)          # metres — discard far points
        self.declare_parameter("min_inlier_ratio", 0.4)    # RANSAC inliers / candidates
        self.declare_parameter("max_translation", 0.5)     # metres per frame
        self.declare_parameter("max_rotation_deg", 30.0)   # degrees per frame
        # Feature tracking params (forwarded to FeatureExtractor)
        self.declare_parameter("max_corners", 200)
        self.declare_parameter("quality_level", 0.5)
        self.declare_parameter("min_distance", 20)
        self.declare_parameter("win_size_w", 14)
        self.declare_parameter("win_size_h", 14)
        self.declare_parameter("max_level", 3)
        self.declare_parameter("max_epipolar_err", 2.0)
        self.declare_parameter("kf_min_translation",  0.05)
        self.declare_parameter("kf_min_rotation_deg", 3.0)
        self.declare_parameter("kf_max_frames",       10)
        self.declare_parameter("pose_cov_pos_base",   0.05)
        self.declare_parameter("pose_cov_ang_base",   0.05)

        config_path = self.get_parameter("config_path").value
        self.min_tracks = self.get_parameter("min_tracks").value
        self.circular_threshold = self.get_parameter("circular_check_threshold").value
        self.max_depth = self.get_parameter("max_depth").value
        self.min_inlier_ratio = self.get_parameter("min_inlier_ratio").value
        self.max_translation = self.get_parameter("max_translation").value
        self.max_rotation_deg = self.get_parameter("max_rotation_deg").value
        self._kf_min_translation  = self.get_parameter("kf_min_translation").value
        self._kf_min_rotation_deg = self.get_parameter("kf_min_rotation_deg").value
        self._kf_max_frames       = self.get_parameter("kf_max_frames").value
        self._pose_cov_pos_base   = self.get_parameter("pose_cov_pos_base").value
        self._pose_cov_ang_base   = self.get_parameter("pose_cov_ang_base").value

        self._T_kf_world_cam0   = None   # world pose at last published keyframe
        self._frames_since_kf   = 0      # frames elapsed since last KF publish
        self._last_inlier_ratio = 1.0    # from most recent successful PnP

        self.load_config(config_path)
        self._setup_camera_params()

        win_w = self.get_parameter("win_size_w").value
        win_h = self.get_parameter("win_size_h").value
        self.extractor = FeatureExtractor(
            max_corners      = self.get_parameter("max_corners").value,
            quality_level    = self.get_parameter("quality_level").value,
            min_distance     = self.get_parameter("min_distance").value,
            win_size         = (win_w, win_h),
            max_level        = self.get_parameter("max_level").value,
            max_epipolar_err = self.get_parameter("max_epipolar_err").value,
        )

        # Feature tracking state
        self._prev_left_features = None
        self._prev_right_features = None

        # World frame = ROS z-up convention (x-fwd, y-left, z-up).
        # The EuRoC body/IMU frame has x≈up, y≈right, z≈fwd (R_b_c0 ≈ R_z(90°)),
        # so a fixed correction rotates the initial world frame so that altitude
        # appears on z and the ESKF gravity vector [0, 0, -9.81] is correct.
        #   body-x (up)    → new z
        #   body-y (right) → new -y  (new y = left)
        #   body-z (fwd)   → new x
        _T_body_to_ros = np.array([[0, 0, 1, 0],
                                   [0,-1, 0, 0],
                                   [1, 0, 0, 0],
                                   [0, 0, 0, 1]], dtype=np.float64)
        self.T_world_cam0 = _T_body_to_ros @ self.T_b_c0

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        self._setup_ros_topics()
        self.get_logger().info("PoseEstimationNode initialized")

    def load_config(self, config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as exc:
            self.get_logger().fatal(
                f"Failed to load config '{config_path}': {exc}. "
                "Pass config_path:=<path> or ensure the package is built."
            )
            raise
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

        # Rectified stereo projection matrices and rectified intrinsics.
        # After rectification distortion is zero and epipolar lines are horizontal.
        R_rel = self.T_c1_c0[:3, :3]
        t_rel = self.T_c1_c0[:3, 3]
        w, h = cam0["resolution"]
        _, _, P0_rect, P1_rect, _, _, _ = cv2.stereoRectify(
            self.cam0_K, self.cam0_dist,
            self.cam1_K, self.cam1_dist,
            (w, h), R_rel, t_rel, alpha=0,
        )
        self.K_rect = P0_rect[:3, :3].copy()
        self.P0 = P0_rect
        self.P1 = P1_rect

    def _setup_ros_topics(self):
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cam0_sub = Subscriber(self, Image, "/cam0/image_rect", qos_profile=qos)
        self.cam1_sub = Subscriber(self, Image, "/cam1/image_rect", qos_profile=qos)

        self.time_sync = ApproximateTimeSynchronizer(
            [self.cam0_sub, self.cam1_sub], queue_size=10, slop=0.1
        )
        self.time_sync.registerCallback(self.stereo_callback)

        self.pose_pub = self.create_publisher(PoseStamped, "/vio/pose", 10)
        self.path_pub = self.create_publisher(Path, "/vio/path", 10)
        self.odom_pub = self.create_publisher(Odometry, "/vio/odometry", 10)
        self.rpy_pub = self.create_publisher(Vector3Stamped, "/vio/rpy", 10)
        self.temporal_viz_pub = self.create_publisher(Image, "/features/temporal_viz", 10)

        self.get_logger().info("ROS topics initialized")

    def _numpy_bgr_to_image_msg(self, img, stamp, frame_id="cam0"):
        """Convert a BGR numpy image to sensor_msgs/Image."""
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        return msg

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

        t_vis = self.extractor.visualize_temporal_tracks(left_img, tracks)
        if t_vis is not None:
            self.temporal_viz_pub.publish(
                self._numpy_bgr_to_image_msg(t_vis, stamp)
            )

        if tracks is None:
            self.get_logger().info(f"ts={ts_ns} | first frame — no pose update")
            return

        if tracks["count"] < self.min_tracks:
            self.get_logger().warn(
                f"ts={ts_ns} | only {tracks['count']} tracks (min={self.min_tracks}), skipping"
            )
            return

        # Images are rectified — keypoints are already in undistorted coordinates
        # --- Triangulate 3D landmarks from the previous stereo pair ---
        pts3d = self._triangulate(tracks["kpts_l_prev"], tracks["kpts_r_prev"])

        # --- PnP: find cam0_curr pose relative to cam0_prev ---
        T_rel = self._solve_pnp(pts3d, tracks["kpts_l_curr"], ts_ns)
        if T_rel is None:
            return

        # T_rel: p_curr = T_rel @ p_prev  →  cam0_curr in world frame:
        self.T_world_cam0 = self.T_world_cam0 @ np.linalg.inv(T_rel)

        if self._T_kf_world_cam0 is None:
            # First successful PnP — always publish
            self._T_kf_world_cam0 = self.T_world_cam0.copy()
            self._frames_since_kf = 0
            self._publish_pose(stamp, ts_ns)
        else:
            delta_T = np.linalg.inv(self._T_kf_world_cam0) @ self.T_world_cam0
            trans     = np.linalg.norm(delta_T[:3, 3])
            cos_angle = np.clip((np.trace(delta_T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            self._frames_since_kf += 1

            if (trans     >= self._kf_min_translation
                    or angle_deg >= self._kf_min_rotation_deg
                    or self._frames_since_kf >= self._kf_max_frames):
                self._T_kf_world_cam0 = self.T_world_cam0.copy()
                self._frames_since_kf = 0
                self._publish_pose(stamp, ts_ns)
            else:
                self.get_logger().debug(
                    f"ts={ts_ns} | skip KF (t={trans:.3f}m r={angle_deg:.1f}° f={self._frames_since_kf})"
                )

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
            self.K_rect,
            None,  # images are rectified — distortion is zero
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

        self._last_inlier_ratio = inlier_ratio

        # Refine on inliers with iterative non-linear optimisation
        inlier_idx = inliers.ravel()
        _, rvec, tvec = cv2.solvePnP(
            pts3d_v[inlier_idx],
            pts2d_v[inlier_idx],
            self.K_rect,
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

        self.get_logger().debug(
            f"ts={ts_ns} | PnP inliers={n_in}/{len(pts3d_v)} ({inlier_ratio:.0%}) "
            f"t={translation:.3f}m rot={angle_deg:.1f}°"
        )

        # Throttled quality warnings — fire at most once per 3 s to avoid spam.
        # Trigger when values are within 30% of their rejection thresholds.
        if inlier_ratio < self.min_inlier_ratio * 1.3:
            self.get_logger().warn(
                f"Low inlier ratio {inlier_ratio:.0%} (threshold {self.min_inlier_ratio:.0%})",
                throttle_duration_sec=3.0,
            )
        if translation > self.max_translation * 0.7:
            self.get_logger().warn(
                f"Large per-frame translation {translation:.3f} m (limit {self.max_translation} m)",
                throttle_duration_sec=3.0,
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
        _ir  = max(self._last_inlier_ratio, 0.1)
        _pv  = (self._pose_cov_pos_base / _ir) ** 2
        _av  = (self._pose_cov_ang_base / _ir) ** 2
        cov  = [0.0] * 36
        cov[0]  = _pv   # x·x
        cov[7]  = _pv   # y·y
        cov[14] = _pv   # z·z
        cov[21] = _av   # roll·roll
        cov[28] = _av   # pitch·pitch
        cov[35] = _av   # yaw·yaw
        odom_msg.pose.covariance = cov
        self.odom_pub.publish(odom_msg)

        roll, pitch, yaw = _rot_to_rpy(R)
        rpy_msg = Vector3Stamped()
        rpy_msg.header.stamp = stamp
        rpy_msg.header.frame_id = "map"
        rpy_msg.vector.x = roll
        rpy_msg.vector.y = pitch
        rpy_msg.vector.z = yaw
        self.rpy_pub.publish(rpy_msg)

        self.get_logger().debug(f"ts={ts_ns} | pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")


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
