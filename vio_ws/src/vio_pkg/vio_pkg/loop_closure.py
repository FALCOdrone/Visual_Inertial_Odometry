import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
import cv2
import numpy as np
import message_filters


class LoopClosureDetector(Node):
    def __init__(self):
        super().__init__("vio_loop_closure")
        self.bridge = CvBridge()

        # --- Camera Intrinsics (must match frontend) ---
        self.fx, self.fy = 458.654, 457.296
        self.cx, self.cy = 367.215, 248.375
        self.K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32
        )
        self.baseline = 0.11

        # --- Camera-to-Body rotation (must match frontend) ---
        self.R_body_cam = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64
        )

        # --- ORB Detector & Matcher ---
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # --- Stereo Matching Parameters ---
        self.lk_params = dict(
            winSize=(21, 21),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # --- Loop Closure Parameters ---
        self.min_loop_gap = 15          # Minimum keyframe index gap to consider loop
        self.min_matches = 25           # Minimum good ORB matches to consider
        self.min_pnp_inliers = 15       # Minimum PnP inliers to confirm loop
        self.match_ratio_thresh = 0.75  # Lowe's ratio test threshold
        self.check_interval = 2         # Check every N keyframes

        # --- Keyframe Database (mirrors mapper) ---
        self.keyframes = []  # list of dicts with pose, descriptors, landmarks, image
        self.sequential_edges = []  # (i, j, relative_pose_4x4)

        # --- ROS Communication ---
        self.loop_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/vio/loop_correction", 10
        )

        # Subscribe to keyframe poses and images
        self.create_subscription(
            PoseStamped, "/vio/keyframe", self.keyframe_pose_callback, 10
        )

        # Subscribe to stereo images for keyframe storage
        left_sub = message_filters.Subscriber(self, Image, "/cam0/image_raw")
        right_sub = message_filters.Subscriber(self, Image, "/cam1/image_raw")
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], 30, 0.05
        )
        self.ts.registerCallback(self.image_callback)

        # Buffer latest images
        self.latest_img_l = None
        self.latest_img_r = None

        self.get_logger().info("Loop Closure Detector Started.")

    def image_callback(self, left_msg, right_msg):
        """Buffer latest stereo images for keyframe descriptor extraction."""
        try:
            self.latest_img_l = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
            self.latest_img_r = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
        except Exception:
            pass

    def keyframe_pose_callback(self, msg):
        """Receive a keyframe pose, extract features, and check for loops."""
        if self.latest_img_l is None:
            return

        # Extract pose
        pos = msg.pose.position
        ori = msg.pose.orientation
        t = np.array([pos.x, pos.y, pos.z])
        q = np.array([ori.x, ori.y, ori.z, ori.w])
        rot = Rot.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t

        # Extract ORB from buffered image
        img_l = self.latest_img_l.copy()
        img_r = self.latest_img_r.copy()
        kp_orb, descriptors = self.orb.detectAndCompute(img_l, None)

        if descriptors is None or len(kp_orb) < 30:
            return

        # Triangulate 3D points in camera frame
        kp_pixels = np.array([k.pt for k in kp_orb], dtype=np.float32).reshape(-1, 1, 2)
        pts_cam, valid = self._triangulate_stereo(img_l, img_r, kp_pixels)

        # Store keyframe
        kf_data = {
            "id": len(self.keyframes),
            "pose": T.copy(),
            "descriptors": descriptors,
            "keypoints_2d": kp_pixels.reshape(-1, 2),
            "landmarks_cam": pts_cam,  # Camera frame
            "valid_3d": valid,
            "image": img_l,
        }

        # Add sequential edge
        if len(self.keyframes) > 0:
            prev_pose = self.keyframes[-1]["pose"]
            rel_pose = np.linalg.inv(prev_pose) @ T
            self.sequential_edges.append(
                (len(self.keyframes) - 1, len(self.keyframes), rel_pose)
            )

        self.keyframes.append(kf_data)

        # --- Check for Loop Closure ---
        kf_idx = len(self.keyframes) - 1
        if kf_idx % self.check_interval == 0 and kf_idx > self.min_loop_gap:
            self._detect_loop(kf_idx)

    def _detect_loop(self, query_idx):
        """Try to detect a loop closure for the given keyframe."""
        query_kf = self.keyframes[query_idx]
        query_desc = query_kf["descriptors"]

        best_match_idx = -1
        best_inliers = 0
        best_rel_pose = None

        # Search through candidate keyframes (skip recent ones)
        for cand_idx in range(0, query_idx - self.min_loop_gap):
            cand_kf = self.keyframes[cand_idx]
            cand_desc = cand_kf["descriptors"]

            # --- ORB Matching with Ratio Test ---
            matches = self.bf_matcher.knnMatch(query_desc, cand_desc, k=2)

            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < self.match_ratio_thresh * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.min_matches:
                continue

            # --- Geometric Verification (PnP) ---
            # Use candidate's 3D points + query's 2D points
            obj_pts = []
            img_pts = []

            for m in good_matches:
                cand_point_idx = m.trainIdx
                query_point_idx = m.queryIdx

                if not cand_kf["valid_3d"][cand_point_idx]:
                    continue

                # Candidate 3D point in camera frame → world frame
                pt_cam = cand_kf["landmarks_cam"][cand_point_idx]
                pt_body = self.R_body_cam @ pt_cam
                pt_world = cand_kf["pose"][:3, :3] @ pt_body + cand_kf["pose"][:3, 3]

                # Transform world point to query camera frame for PnP
                # world → query body → query camera
                pt_query_body = (
                    query_kf["pose"][:3, :3].T @ (pt_world - query_kf["pose"][:3, 3])
                )
                pt_query_cam = self.R_body_cam.T @ pt_query_body

                obj_pts.append(pt_query_cam)
                img_pts.append(query_kf["keypoints_2d"][query_point_idx])

            if len(obj_pts) < self.min_pnp_inliers:
                continue

            obj_pts = np.array(obj_pts, dtype=np.float64)
            img_pts = np.array(img_pts, dtype=np.float64).reshape(-1, 1, 2)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts,
                img_pts,
                self.K,
                None,
                reprojectionError=3.0,
                iterationsCount=200,
                confidence=0.999,
                flags=cv2.SOLVEPNP_SQPNP,
            )

            if success and inliers is not None and len(inliers) > best_inliers:
                best_inliers = len(inliers)
                best_match_idx = cand_idx
                # Compute relative pose in world frame
                R_pnp, _ = cv2.Rodrigues(rvec)
                T_rel = np.eye(4)
                T_rel[:3, :3] = R_pnp
                T_rel[:3, 3] = tvec.flatten()
                best_rel_pose = T_rel

        # --- If loop found, run pose graph optimization ---
        if best_match_idx >= 0 and best_inliers >= self.min_pnp_inliers:
            self.get_logger().info(
                f"LOOP CLOSURE DETECTED: KF {query_idx} ↔ KF {best_match_idx} "
                f"({best_inliers} inliers)"
            )

            # Compute the corrected pose via pose graph optimization
            corrected_poses = self._optimize_pose_graph(
                query_idx, best_match_idx, best_rel_pose
            )

            if corrected_poses is not None:
                self._publish_correction(corrected_poses, query_idx)

    def _optimize_pose_graph(self, loop_query_idx, loop_match_idx, loop_rel_pose):
        """
        Simple pose graph optimization using scipy.

        Optimizes over keyframe positions (translation only for simplicity)
        with sequential VO constraints + the loop closure constraint.
        """
        n = len(self.keyframes)
        if n < 3:
            return None

        # Initial guess: current keyframe positions (flatten to 3N vector)
        x0 = np.zeros(n * 3)
        for i, kf in enumerate(self.keyframes):
            x0[i * 3 : i * 3 + 3] = kf["pose"][:3, 3]

        # Build edges: sequential + loop
        edges = []

        # Sequential edges from VO
        for i, j, T_rel in self.sequential_edges:
            if i < n and j < n:
                delta_t = T_rel[:3, 3]
                edges.append((i, j, delta_t, 1.0))  # weight = 1.0

        # Loop closure edge
        cand_pose = self.keyframes[loop_match_idx]["pose"]
        query_pose = self.keyframes[loop_query_idx]["pose"]
        # The loop constraint says query should be at cand_pose + some offset
        loop_delta = (
            np.linalg.inv(cand_pose) @ query_pose
        )[:3, 3]
        edges.append(
            (loop_match_idx, loop_query_idx, loop_delta, 5.0)  # higher weight
        )

        def residuals(x):
            res = []
            for i, j, delta_t, weight in edges:
                pi = x[i * 3 : i * 3 + 3]
                pj = x[j * 3 : j * 3 + 3]
                # Predicted relative translation
                predicted_delta = pj - pi
                err = weight * (predicted_delta - delta_t)
                res.extend(err)
            # Pin first keyframe to origin
            res.extend(10.0 * x[0:3])
            return np.array(res)

        result = least_squares(residuals, x0, method="lm", max_nfev=500)

        if not result.success:
            self.get_logger().warn("Pose graph optimization failed to converge.")
            return None

        # Extract optimized poses
        x_opt = result.x
        corrected_poses = []
        for i in range(n):
            T_new = self.keyframes[i]["pose"].copy()
            T_new[:3, 3] = x_opt[i * 3 : i * 3 + 3]
            corrected_poses.append(T_new)

        # Update stored keyframe poses
        for i, kf in enumerate(self.keyframes):
            kf["pose"] = corrected_poses[i]

        # Recompute sequential edges with corrected poses
        self.sequential_edges = []
        for i in range(len(self.keyframes) - 1):
            rel = np.linalg.inv(self.keyframes[i]["pose"]) @ self.keyframes[i + 1]["pose"]
            self.sequential_edges.append((i, i + 1, rel))

        self.get_logger().info(
            f"Pose graph optimized: cost {result.cost:.4f}, {result.nfev} iterations"
        )
        return corrected_poses

    def _publish_correction(self, corrected_poses, query_idx):
        """Publish the corrected latest pose as a loop correction."""
        corrected = corrected_poses[query_idx]

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"

        msg.pose.pose.position.x = float(corrected[0, 3])
        msg.pose.pose.position.y = float(corrected[1, 3])
        msg.pose.pose.position.z = float(corrected[2, 3])

        q = Rot.from_matrix(corrected[:3, :3]).as_quat()
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])

        # Low covariance = high confidence in the correction
        msg.pose.covariance[0] = 0.01
        msg.pose.covariance[7] = 0.01
        msg.pose.covariance[14] = 0.01
        msg.pose.covariance[21] = 0.02
        msg.pose.covariance[28] = 0.02
        msg.pose.covariance[35] = 0.02

        self.loop_pub.publish(msg)
        self.get_logger().info("Published loop closure correction.")

    def _triangulate_stereo(self, img_l, img_r, kp_l):
        """Triangulate 3D points from stereo pair. Returns points in camera frame."""
        kp_l = kp_l.astype(np.float32)
        kp_r, st, _ = cv2.calcOpticalFlowPyrLK(
            img_l, img_r, kp_l, None, **self.lk_params
        )

        n_points = len(kp_l)
        pts_3d = np.zeros((n_points, 3), dtype=np.float32)
        valid = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            if st[i][0] != 1:
                continue
            pl = kp_l[i].ravel()
            pr = kp_r[i].ravel()
            d = pl[0] - pr[0]

            if d > 1.0:
                z = (self.fx * self.baseline) / d
                x = (pl[0] - self.cx) * z / self.fx
                y = (pl[1] - self.cy) * z / self.fy
                if z < 30.0:
                    pts_3d[i] = [x, y, z]
                    valid[i] = True

        return pts_3d, valid


def main(args=None):
    rclpy.init(args=args)
    node = LoopClosureDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
