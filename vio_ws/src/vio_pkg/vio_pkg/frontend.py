import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from sensor_msgs.msg import Image  # type: ignore
from geometry_msgs.msg import PoseWithCovarianceStamped  # type: ignore
import message_filters  # type: ignore
from cv_bridge import CvBridge  # type: ignore
import cv2  # type: ignore
import numpy as np  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
import traceback  # type: ignore


class VIOFrontend(Node):
    def __init__(self):
        super().__init__("vio_frontend")
        self.bridge = CvBridge()

        # --- Camera Intrinsics (EuRoC MAV Dataset Approx) ---
        self.fx, self.fy = 458.654, 457.296
        self.cx, self.cy = 367.215, 248.375
        self.K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32
        )
        self.baseline = 0.11  # meters

        # --- State Variables ---
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_pts_3d = None

        # --- Feature Detector Parameters ---
        self.feature_params = dict(
            maxCorners=800, qualityLevel=0.005, minDistance=10, blockSize=7
        )

        # --- Optical Flow Parameters ---
        self.lk_params = dict(
            winSize=(21, 21),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # --- ROS Communication ---
        self.vo_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/vio/visual_odom", 10
        )
        self.feature_count_pub = self.create_publisher(
            Image, "/vio/feature_count_image", 10
        )
        self.tracked_keypoints_pub = self.create_publisher(
            Image, "/vio/tracked_keypoints_image", 10
        )

        left_sub = message_filters.Subscriber(self, Image, "/cam0/image_raw")
        right_sub = message_filters.Subscriber(self, Image, "/cam1/image_raw")

        # Increased slop slightly to handle network jitter better
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], 10, 0.02
        )
        self.ts.registerCallback(self.stereo_callback)

        self.get_logger().info("VIO Frontend Started. Waiting for images...")

        self.global_transform = np.eye(4)

        # Camera-to-Body rotation (EuRoC cam0 → body/IMU frame)
        # Camera: X-right, Y-down, Z-forward
        # Body:   X-up,    Y-right, Z-forward  (EuRoC IMU)
        # Maps:   body_X = -cam_Y, body_Y = cam_X, body_Z = cam_Z
        self.R_body_cam = np.array([
            [ 0, -1,  0],
            [ 1,  0,  0],
            [ 0,  0,  1]
        ], dtype=np.float64)
        self.T_body_cam = np.eye(4)
        self.T_body_cam[:3, :3] = self.R_body_cam
        self.T_body_cam_inv = np.linalg.inv(self.T_body_cam)

    def _initialize_vo(self, img_l, img_r):
        """Handles the first frame to initialize the VO pipeline."""
        self.get_logger().info("Initializing VIO Frontend...")
        self.prev_gray = img_l
        self.prev_keypoints = self._detect_features(img_l)

        if self.prev_keypoints is None or len(self.prev_keypoints) == 0:
            self.prev_gray = None
            raise ValueError("Initialization failed: No keypoints detected.")

        valid_kp, self.prev_pts_3d = self.triangulate_stereo(
            img_l, img_r, self.prev_keypoints
        )
        # Stricter check on initialization
        if valid_kp is None or len(valid_kp) < 30:
            self.prev_gray = None
            raise ValueError(
                f"Initialization failed: Not enough stereo matches ({len(valid_kp) if valid_kp is not None else 0})."
            )

        self.prev_keypoints = valid_kp
        self.get_logger().info("Initialization successful.")
        return f"Initialized with {len(self.prev_keypoints)} points."

    def _track_features(self, current_gray):
        """Tracks features with forward-backward consistency check."""
        # Forward tracking: prev → curr
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, current_gray, self.prev_keypoints, None, **self.lk_params
        )

        if p1 is None:
            raise ValueError("Optical Flow failed.")

        # Backward tracking: curr → prev (consistency check)
        p0_back, st2, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, self.prev_gray, p1, None, **self.lk_params
        )

        # Compute round-trip error
        fb_err = np.linalg.norm(
            (self.prev_keypoints - p0_back).reshape(-1, 2), axis=1
        )

        # Valid = tracked in both directions AND round-trip error < 1 pixel
        st = (st1.flatten() == 1) & (st2.flatten() == 1) & (fb_err < 1.0)

        if np.sum(st) < 10:
            raise ValueError(f"Tracking lost ({np.sum(st)} points after FB check).")

        good_new_2d = p1[st].reshape(-1, 1, 2)
        good_old_2d = self.prev_keypoints[st].reshape(-1, 1, 2)
        good_old_3d = self.prev_pts_3d[st]

        return good_new_2d, good_old_2d, good_old_3d

    def _estimate_motion(self, good_old_3d, good_new_2d):
        """Estimates camera motion using PnP RANSAC."""
        MIN_PNP_POINTS = 15
        if len(good_new_2d) <= MIN_PNP_POINTS:
            raise ValueError(
                f"Not enough points for PnP ({len(good_new_2d)} < {MIN_PNP_POINTS})."
            )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            good_old_3d,
            good_new_2d,
            self.K,
            None,
            reprojectionError=1.5,  # Tight threshold for high-quality inliers
            iterationsCount=300,    # More iterations for better RANSAC solution
            confidence=0.999,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not success:
            raise ValueError("PnP RANSAC failed.")

        if inliers is None or len(inliers) < MIN_PNP_POINTS:
            raise ValueError(
                f"PnP success, but too few inliers ({len(inliers) if inliers is not None else 0})."
            )

        inliers_flat = inliers.flatten()
        pnp_new_2d = good_new_2d[inliers_flat]
        pnp_old_3d = good_old_3d[inliers_flat]

        # Return inliers count for covariance estimation
        return rvec, tvec, pnp_new_2d, pnp_old_3d, len(inliers)

    def _maintain_features(self, current_gray, tracked_kp):
        """Replenishes features if the count drops below a threshold."""
        num_tracked_features = len(tracked_kp)
        MIN_FEATURES = 400

        new_features_kp = None
        if num_tracked_features < MIN_FEATURES:
            mask = np.full(current_gray.shape, 255, dtype=np.uint8)
            for pt in tracked_kp:
                x, y = pt.ravel()
                cv2.circle(mask, (int(x), int(y)), 15, 0, -1)

            new_features_kp = self._detect_features(current_gray, mask)

        return new_features_kp

    def stereo_callback(self, left_msg, right_msg):
        vis_img = None
        status_text = ""
        status_color = (0, 0, 255)

        try:
            img_l = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
            img_r = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
            vis_img = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)

            if self.prev_gray is None:
                status_text = self._initialize_vo(img_l, img_r)
                status_color = (0, 255, 0)
                raise StopIteration(status_text)

            # --- Tracking & Motion Estimation ---
            good_new_2d, good_old_2d, good_old_3d = self._track_features(img_l)
            rvec, tvec, pnp_new_2d, pnp_old_3d, inlier_count = self._estimate_motion(
                good_old_3d, good_new_2d
            )

            # --- Publish Odometry with Dynamic Covariance ---
            self.publish_visual_odom(rvec, tvec, left_msg.header, inlier_count)

            # --- Propagation & Maintenance ---
            R_mat, _ = cv2.Rodrigues(rvec)
            fallback_3d = (R_mat @ pnp_old_3d.T + tvec).T

            num_tracked_features = len(pnp_new_2d)
            new_features_kp = self._maintain_features(img_l, pnp_new_2d)

            current_kp = pnp_new_2d
            if new_features_kp is not None and len(new_features_kp) > 0:
                current_kp = np.concatenate((pnp_new_2d, new_features_kp), axis=0)
                nan_padding = np.full(
                    (len(new_features_kp), 3), np.nan, dtype=np.float32
                )
                fallback_3d = np.concatenate((fallback_3d, nan_padding), axis=0)

            MAX_FEATURES = 500
            current_kp = current_kp[:MAX_FEATURES]
            if fallback_3d is not None:
                fallback_3d = fallback_3d[:MAX_FEATURES]

            valid_kp, next_pts_3d = self.triangulate_stereo(
                img_l, img_r, current_kp, fallback_3d
            )

            if valid_kp is None or len(valid_kp) == 0:
                raise ValueError("Triangulation failed for all points.")

            self.prev_keypoints = valid_kp
            self.prev_pts_3d = next_pts_3d
            self.prev_gray = img_l
            status_text = f"Inliers: {inlier_count} | Tracked: {len(valid_kp)}"
            status_color = (0, 255, 0)

            # --- Visualization (Restored to Original Style) ---
            # 1. Tracked Keypoints Image (Lines + Red Dots)
            for new, old in zip(good_new_2d, good_old_2d):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(vis_img, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.circle(
                    vis_img, (int(a), int(b)), 3, (0, 0, 255), -1
                )  # Restored red dots
            self.tracked_keypoints_pub.publish(
                self.bridge.cv2_to_imgmsg(vis_img, "bgr8", header=left_msg.header)
            )

            # 2. Feature Count Image (Green vs Blue dots + Text overlay)
            vis_feature_img = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            for pt in pnp_new_2d:
                cv2.circle(
                    vis_feature_img,
                    (int(pt.ravel()[0]), int(pt.ravel()[1])),
                    3,
                    (0, 255, 0),
                    -1,
                )
            if new_features_kp is not None:
                for pt in new_features_kp:
                    # Blue dots for new features
                    cv2.circle(
                        vis_feature_img,
                        (int(pt.ravel()[0]), int(pt.ravel()[1])),
                        3,
                        (255, 0, 0),
                        -1,
                    )

            new_count = len(new_features_kp) if new_features_kp is not None else 0
            count_text = f"Tracked: {num_tracked_features}, New: {new_count}"
            cv2.putText(
                vis_feature_img,
                count_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            self.feature_count_pub.publish(
                self.bridge.cv2_to_imgmsg(
                    vis_feature_img, "bgr8", header=left_msg.header
                )
            )

        except StopIteration as e:
            status_text = str(e)
        except ValueError as e:
            status_text = str(e)
            self.get_logger().warn(status_text)
            self.prev_gray = None
        except Exception as e:
            status_text = f"Error: {e}"
            self.get_logger().error(
                f"Error in processing: {e}\n{traceback.format_exc()}"
            )
            self.prev_gray = None

        if vis_img is not None:
            # Keep original font scale and color logic for the debug window
            cv2.putText(
                vis_img,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.imshow("VIO Frontend", vis_img)
            cv2.waitKey(1)

    def _detect_features(self, image, mask=None):
        corners = cv2.goodFeaturesToTrack(image, mask=mask, **self.feature_params)
        if corners is not None:
            return corners.reshape(-1, 1, 2)
        return None

    def triangulate_stereo(self, img_l, img_r, kp_l, fallback_3d=None):
        if kp_l is None or len(kp_l) == 0:
            return None, np.zeros((0, 3))

        kp_l = kp_l.astype(np.float32)
        kp_r, st, _ = cv2.calcOpticalFlowPyrLK(
            img_l, img_r, kp_l, None, **self.lk_params
        )

        points3d = []
        good_keypoints = []

        for i, (pl, pr) in enumerate(zip(kp_l, kp_r)):
            valid_point = False
            if st[i][0] == 1:
                pl_flat = pl.ravel()
                pr_flat = pr.ravel()
                d = pl_flat[0] - pr_flat[0]

                if d > 1.0:
                    z = (self.fx * self.baseline) / d
                    x = (pl_flat[0] - self.cx) * z / self.fx
                    y = (pl_flat[1] - self.cy) * z / self.fy
                    points3d.append([x, y, z])
                    good_keypoints.append(pl)
                    valid_point = True

            if not valid_point and fallback_3d is not None:
                if not np.isnan(fallback_3d[i, 0]):
                    points3d.append(fallback_3d[i])
                    good_keypoints.append(pl)

        if not good_keypoints:
            return None, np.zeros((0, 3))

        return np.array(good_keypoints, dtype=np.float32).reshape(-1, 1, 2), np.array(
            points3d, dtype=np.float32
        )

    def publish_visual_odom(self, rvec, tvec, header, inlier_count):
        # PnP gives R,t: p_current = R @ p_prev + t (prev 3D → current camera)
        # Invert to get current camera pose in previous frame
        R_mat, _ = cv2.Rodrigues(rvec)
        R_inv = R_mat.T
        t_inv = -R_inv @ tvec

        # --- MOTION GATING ---
        step_distance = np.linalg.norm(t_inv)
        if step_distance > 1.0:
            self.get_logger().warn(
                f"Huge jump detected ({step_distance:.2f}m). Ignoring frame."
            )
            return

        step_transform = np.eye(4)
        step_transform[:3, :3] = R_inv
        step_transform[:3, 3] = t_inv.flatten()

        self.global_transform = self.global_transform @ step_transform

        # Transform from camera frame to body/IMU frame
        global_body = self.T_body_cam @ self.global_transform @ self.T_body_cam_inv
        global_t = global_body[:3, 3]
        global_R = global_body[:3, :3]
        q = R.from_matrix(global_R).as_quat()

        msg = PoseWithCovarianceStamped()
        msg.header = header
        msg.header.frame_id = "odom"

        msg.pose.pose.position.x = float(global_t[0])
        msg.pose.pose.position.y = float(global_t[1])
        msg.pose.pose.position.z = float(global_t[2])
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])

        # --- DYNAMIC COVARIANCE ---
        # Lower covariance = trust VO more. Scaled by inlier count.
        variance_val = 1.0 / (inlier_count + 1) * 0.1
        variance_val = max(variance_val, 0.0001)  # Floor
        variance_val = min(variance_val, 0.5)      # Cap

        # Fill diagonal for x, y, z
        msg.pose.covariance[0] = variance_val
        msg.pose.covariance[7] = variance_val
        msg.pose.covariance[14] = variance_val
        # Orientation covariance (roll, pitch, yaw)
        orientation_var = variance_val * 2.0
        msg.pose.covariance[21] = orientation_var
        msg.pose.covariance[28] = orientation_var
        msg.pose.covariance[35] = orientation_var

        self.vo_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VIOFrontend()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
