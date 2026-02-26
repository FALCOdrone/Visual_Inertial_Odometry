# =============================================================================
# VIO Frontend (RAFT) – Stereo visual odometry via RAFT optical flow + PnP
# Pipeline: detect features → track (RAFT dense flow) → triangulate (stereo)
#           → estimate motion (PnP RANSAC) → publish pose to backend EKF
#
# Multi-frame: keeps a sliding window of k previous frames. Tracks are
# established from ALL buffered frames to the current frame, giving PnP
# a richer set of 3D-2D correspondences.
# =============================================================================

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
import time  # type: ignore
from collections import deque

import torch
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights


class VIOFrontendRAFT(Node):
    def __init__(self):
        super().__init__("vio_frontend_raft")
        self.bridge = CvBridge()

        # ==================== Device Selection ====================
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.get_logger().info("RAFT running on CUDA GPU.")
        else:
            self.device = torch.device("cpu")
            self.get_logger().warn(
                "RAFT running on CPU – expect ~200 ms/frame. "
                "Install CUDA for real-time performance."
            )

        # ==================== RAFT Model ====================
        weights = Raft_Small_Weights.DEFAULT
        self.raft_model = raft_small(weights=weights).to(self.device)
        self.raft_model.eval()
        # Input transforms recommended by torchvision
        self.raft_transforms = weights.transforms()
        self.get_logger().info("RAFT-Small model loaded.")

        # ==================== Camera Intrinsics (EuRoC cam0) ====================
        # TUNABLE: Must match the camera calibration of your dataset/sensor.
        self.fx, self.fy = 458.654, 457.296   # focal lengths (px)
        self.cx, self.cy = 367.215, 248.375   # principal point  (px)
        self.K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32
        )
        self.baseline = 0.11  # stereo baseline (m) – distance between cam0 & cam1

        # ==================== State Variables ====================
        self.prev_gray = None       # previous left image (grayscale)
        self.prev_keypoints = None  # previous 2D keypoints (Nx1x2)
        self.prev_pts_3d = None     # previous 3D points   (Nx3)

        # ==================== TUNABLE: Multi-Frame Buffer ====================
        # K_FRAMES: number of past frames to keep for multi-frame tracking.
        # More frames → more 3D-2D correspondences for PnP (more robust)
        # but also more RAFT inference calls per frame (slower).
        # Recommended range: 3–10.
        self.K_FRAMES = 5
        self.frame_buffer = deque(maxlen=self.K_FRAMES)

        # ==================== TUNABLE: Feature Detector ====================
        # goodFeaturesToTrack (Shi-Tomasi corners)
        # maxCorners   – max features per frame  (more → denser but slower)
        # qualityLevel – corner quality threshold (lower → more features)
        # minDistance   – min pixel spacing        (higher → more spread out)
        # blockSize    – neighbourhood size        (larger → smoother response)
        self.feature_params = dict(
            maxCorners=800, qualityLevel=0.01, minDistance=10, blockSize=7
        )

        # ==================== TUNABLE: LK params (stereo matching only) ====================
        # LK is still used for LEFT→RIGHT stereo matching (triangulation).
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

        # TUNABLE: slop (0.02 s) – max time difference between left/right frames
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], 10, 0.02
        )
        self.ts.registerCallback(self.stereo_callback)

        self.get_logger().info("VIO Frontend (RAFT) Started. Waiting for images...")

        self.global_transform = np.eye(4)  # accumulated camera pose (cam frame)

        # ==================== Frame Transform: Camera → Body ====================
        # body_X = cam_Z, body_Y = -cam_X, body_Z = -cam_Y
        self.R_body_cam = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ], dtype=np.float64)
        self.T_body_cam = np.eye(4)
        self.T_body_cam[:3, :3] = self.R_body_cam
        self.T_body_cam_inv = np.linalg.inv(self.T_body_cam)

    # ======================== RAFT Inference ========================
    def _run_raft(self, gray_prev, gray_curr):
        """
        Run RAFT optical flow from gray_prev to gray_curr.
        Returns dense flow field as numpy array (H, W, 2).
        """
        # RAFT expects 3-channel uint8 tensors [B, 3, H, W]
        img1_rgb = cv2.cvtColor(gray_prev, cv2.COLOR_GRAY2RGB)
        img2_rgb = cv2.cvtColor(gray_curr, cv2.COLOR_GRAY2RGB)

        # Convert to tensor [H, W, C] → [C, H, W] → float [0, 1]
        t1 = torch.from_numpy(img1_rgb).permute(2, 0, 1).float()
        t2 = torch.from_numpy(img2_rgb).permute(2, 0, 1).float()

        # Apply recommended transforms (normalisation)
        t1, t2 = self.raft_transforms(t1, t2)

        # Add batch dimension
        t1 = t1.unsqueeze(0).to(self.device)
        t2 = t2.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # RAFT returns a list of flow predictions (one per iteration);
            # last element is the most refined.
            flow_preds = self.raft_model(t1, t2)
            flow = flow_preds[-1]  # [1, 2, H, W]

        # Convert to numpy (H, W, 2)
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
        return flow_np

    def _sample_flow_at_keypoints(self, flow, keypoints):
        """
        Given a dense flow field (H, W, 2) and keypoints (Nx1x2),
        sample the flow at each keypoint via bilinear interpolation
        and return displaced keypoint positions (Nx1x2).
        """
        H, W, _ = flow.shape
        kp = keypoints.reshape(-1, 2).astype(np.float32)

        # Bilinear sampling using cv2.remap
        map_x = kp[:, 0]
        map_y = kp[:, 1]

        # Sample flow_x and flow_y channels at keypoint locations
        flow_x = cv2.remap(
            flow[:, :, 0].astype(np.float32),
            map_x.reshape(1, -1),
            map_y.reshape(1, -1),
            cv2.INTER_LINEAR,
        ).flatten()
        flow_y = cv2.remap(
            flow[:, :, 1].astype(np.float32),
            map_x.reshape(1, -1),
            map_y.reshape(1, -1),
            cv2.INTER_LINEAR,
        ).flatten()

        new_kp = np.stack([kp[:, 0] + flow_x, kp[:, 1] + flow_y], axis=-1)

        # Clamp to image bounds
        new_kp[:, 0] = np.clip(new_kp[:, 0], 0, W - 1)
        new_kp[:, 1] = np.clip(new_kp[:, 1], 0, H - 1)

        return new_kp.reshape(-1, 1, 2)

    # ======================== Pipeline Stage 0: Init ========================
    def _initialize_vo(self, img_l, img_r):
        """First-frame bootstrap: detect features, triangulate stereo 3D points."""
        self.get_logger().info("Initializing VIO Frontend (RAFT)...")
        self.prev_gray = img_l
        self.prev_keypoints = self._detect_features(img_l)

        if self.prev_keypoints is None or len(self.prev_keypoints) == 0:
            self.prev_gray = None
            raise ValueError("Initialization failed: No keypoints detected.")

        valid_kp, self.prev_pts_3d = self.triangulate_stereo(
            img_l, img_r, self.prev_keypoints
        )
        # TUNABLE: min stereo matches to accept initialization (default 30)
        if valid_kp is None or len(valid_kp) < 30:
            self.prev_gray = None
            raise ValueError(
                f"Initialization failed: Not enough stereo matches ({len(valid_kp) if valid_kp is not None else 0})."
            )

        self.prev_keypoints = valid_kp
        self.prev_pts_3d = self.prev_pts_3d

        # Seed the frame buffer with the first frame
        self.frame_buffer.clear()
        self.frame_buffer.append({
            'gray': img_l.copy(),
            'keypoints': valid_kp.copy(),
            'pts_3d': self.prev_pts_3d.copy(),
        })

        self.get_logger().info("Initialization successful (RAFT).")
        return f"Initialized with {len(self.prev_keypoints)} points."

    # ===================== Pipeline Stage 1: Track (RAFT) =====================
    def _track_features(self, current_gray):
        """
        RAFT-based tracking with forward-backward consistency check.
        Tracks from ONLY the immediately previous frame (like original LK).
        Multi-frame tracking is handled separately in the callback.
        """
        t0 = time.perf_counter()

        # Forward pass: prev → curr (RAFT)
        flow_fwd = self._run_raft(self.prev_gray, current_gray)
        new_kp = self._sample_flow_at_keypoints(flow_fwd, self.prev_keypoints)

        # Backward pass: curr → prev (consistency check)
        flow_bwd = self._run_raft(current_gray, self.prev_gray)
        back_kp = self._sample_flow_at_keypoints(flow_bwd, new_kp)

        # Round-trip reprojection error
        fb_err = np.linalg.norm(
            (self.prev_keypoints - back_kp).reshape(-1, 2), axis=1
        )

        # TUNABLE: fb_err < 2.0 px – forward-backward consistency threshold
        # Slightly more permissive than LK (1.0) because RAFT sub-pixel
        # precision may differ between forward and backward at boundaries.
        st = fb_err < 2.0

        elapsed = (time.perf_counter() - t0) * 1000
        self.get_logger().info(
            f"RAFT track: {np.sum(st)}/{len(st)} survived FB check "
            f"({elapsed:.0f} ms)"
        )

        # TUNABLE: minimum surviving tracks to continue (default 10)
        if np.sum(st) < 10:
            raise ValueError(f"Tracking lost ({np.sum(st)} points after FB check).")

        good_new_2d = new_kp[st].reshape(-1, 1, 2)
        good_old_2d = self.prev_keypoints[st].reshape(-1, 1, 2)
        good_old_3d = self.prev_pts_3d[st]

        return good_new_2d, good_old_2d, good_old_3d

    # ============= Pipeline Stage 1b: Multi-Frame Track (RAFT) =============
    def _track_from_buffer(self, current_gray):
        """
        Track features from ALL buffered frames to the current frame.
        Merges correspondences to give PnP a richer set of 3D-2D matches.
        Returns (merged_new_2d, merged_old_3d) after NMS de-duplication.
        """
        if len(self.frame_buffer) <= 1:
            return None, None

        all_new_2d = []
        all_old_3d = []

        # Track from each buffered frame (skip most recent, handled by _track_features)
        for i, entry in enumerate(list(self.frame_buffer)[:-1]):
            buf_gray = entry['gray']
            buf_kp = entry['keypoints']
            buf_3d = entry['pts_3d']

            if buf_kp is None or len(buf_kp) == 0:
                continue

            try:
                flow_fwd = self._run_raft(buf_gray, current_gray)
                new_kp = self._sample_flow_at_keypoints(flow_fwd, buf_kp)

                # Quick backward check
                flow_bwd = self._run_raft(current_gray, buf_gray)
                back_kp = self._sample_flow_at_keypoints(flow_bwd, new_kp)
                fb_err = np.linalg.norm(
                    (buf_kp - back_kp).reshape(-1, 2), axis=1
                )

                # More permissive for older frames (larger motion)
                threshold = 2.0 + 0.5 * (len(self.frame_buffer) - 1 - i)
                st = fb_err < threshold

                if np.sum(st) > 0:
                    all_new_2d.append(new_kp[st].reshape(-1, 2))
                    all_old_3d.append(buf_3d[st])
            except Exception:
                continue

        if not all_new_2d:
            return None, None

        merged_2d = np.concatenate(all_new_2d, axis=0)
        merged_3d = np.concatenate(all_old_3d, axis=0)

        # NMS-style de-duplication: keep only one point per 5px neighbourhood
        if len(merged_2d) > 0:
            merged_2d, merged_3d = self._nms_keypoints(merged_2d, merged_3d, radius=5.0)

        return merged_2d.reshape(-1, 1, 2), merged_3d

    def _nms_keypoints(self, kp_2d, pts_3d, radius=5.0):
        """
        Non-maximum suppression on 2D keypoints.
        Keeps the first occurrence; removes later points within `radius` px.
        """
        keep = np.ones(len(kp_2d), dtype=bool)
        for i in range(len(kp_2d)):
            if not keep[i]:
                continue
            dists = np.linalg.norm(kp_2d[i + 1:] - kp_2d[i], axis=1)
            too_close = np.where(dists < radius)[0] + i + 1
            keep[too_close] = False
        return kp_2d[keep], pts_3d[keep]

    # ================== Pipeline Stage 2: Motion Estimation ==================
    def _estimate_motion(self, good_old_3d, good_new_2d):
        """3D-2D PnP RANSAC → camera-frame R, t."""
        # TUNABLE: min 3D-2D correspondences required for PnP
        MIN_PNP_POINTS = 15
        if len(good_new_2d) <= MIN_PNP_POINTS:
            raise ValueError(
                f"Not enough points for PnP ({len(good_new_2d)} < {MIN_PNP_POINTS})."
            )

        # TUNABLE: reprojectionError – RANSAC inlier threshold (px)
        #          iterationsCount   – RANSAC iterations
        #          confidence        – RANSAC success probability
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            good_old_3d,
            good_new_2d,
            self.K,
            None,
            reprojectionError=1.5,
            iterationsCount=300,
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

        return rvec, tvec, pnp_new_2d, pnp_old_3d, len(inliers)

    # ================ Pipeline Stage 3: Feature Maintenance ================
    def _maintain_features(self, current_gray, tracked_kp):
        """Detect new features when tracked count drops below threshold."""
        num_tracked_features = len(tracked_kp)
        # TUNABLE: replenish features when count falls below this
        MIN_FEATURES = 400

        new_features_kp = None
        if num_tracked_features < MIN_FEATURES:
            # Mask out existing feature neighbourhoods (15 px radius)
            mask = np.full(current_gray.shape, 255, dtype=np.uint8)
            for pt in tracked_kp:
                x, y = pt.ravel()
                cv2.circle(mask, (int(x), int(y)), 15, 0, -1)

            new_features_kp = self._detect_features(current_gray, mask)

        return new_features_kp

    # ==================== Main Stereo Callback ====================
    def stereo_callback(self, left_msg, right_msg):
        """Called on each synchronized stereo pair. Runs full VO pipeline."""
        vis_img = None
        status_text = ""
        status_color = (0, 0, 255)  # red = error; overwritten to green on success

        try:
            img_l = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
            img_r = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
            vis_img = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)

            # --- First frame: bootstrap ---
            if self.prev_gray is None:
                status_text = self._initialize_vo(img_l, img_r)
                status_color = (0, 255, 0)
                raise StopIteration(status_text)

            # --- Stage 1: Track features (primary: prev frame) ---
            good_new_2d, good_old_2d, good_old_3d = self._track_features(img_l)

            # --- Stage 1b: Multi-frame augmentation ---
            # Merge correspondences from older buffered frames for richer PnP
            buf_new_2d, buf_old_3d = self._track_from_buffer(img_l)
            if buf_new_2d is not None and len(buf_new_2d) > 0:
                combined_new_2d = np.concatenate(
                    [good_new_2d, buf_new_2d], axis=0
                )
                combined_old_3d = np.concatenate(
                    [good_old_3d, buf_old_3d], axis=0
                )
                self.get_logger().info(
                    f"Multi-frame: +{len(buf_new_2d)} correspondences "
                    f"(total {len(combined_new_2d)})"
                )
            else:
                combined_new_2d = good_new_2d
                combined_old_3d = good_old_3d

            # --- Stage 2: Motion estimation (PnP) ---
            rvec, tvec, pnp_new_2d, pnp_old_3d, inlier_count = (
                self._estimate_motion(combined_old_3d, combined_new_2d)
            )

            # --- Publish pose to backend EKF ---
            self.publish_visual_odom(rvec, tvec, left_msg.header, inlier_count)

            # --- Stage 3: Propagate 3D points & replenish features ---
            # Use only the primary-frame tracked points for state propagation
            R_mat, _ = cv2.Rodrigues(rvec)
            primary_pnp_mask = np.arange(len(good_new_2d))
            # Filter to only primary-frame inliers for propagation
            primary_new_2d = good_new_2d
            primary_old_3d = good_old_3d
            fallback_3d = (R_mat @ primary_old_3d.T + tvec).T

            num_tracked_features = len(primary_new_2d)
            new_features_kp = self._maintain_features(img_l, primary_new_2d)

            current_kp = primary_new_2d
            if new_features_kp is not None and len(new_features_kp) > 0:
                current_kp = np.concatenate(
                    (primary_new_2d, new_features_kp), axis=0
                )
                nan_padding = np.full(
                    (len(new_features_kp), 3), np.nan, dtype=np.float32
                )
                fallback_3d = np.concatenate((fallback_3d, nan_padding), axis=0)

            MAX_FEATURES = 150
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

            # --- Push frame into multi-frame buffer ---
            self.frame_buffer.append({
                'gray': img_l.copy(),
                'keypoints': valid_kp.copy(),
                'pts_3d': next_pts_3d.copy(),
            })

            status_text = (
                f"Inliers: {inlier_count} | Tracked: {len(valid_kp)} | "
                f"Buffer: {len(self.frame_buffer)}"
            )
            status_color = (0, 255, 0)

            # ---- Visualization: tracked keypoints (green lines + red dots) ----
            for new, old in zip(good_new_2d, good_old_2d):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(vis_img, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.circle(vis_img, (int(a), int(b)), 3, (0, 0, 255), -1)

            # Draw multi-frame matches in cyan
            if buf_new_2d is not None:
                for pt in buf_new_2d:
                    x, y = pt.ravel()
                    cv2.circle(
                        vis_img, (int(x), int(y)), 4, (255, 255, 0), 1
                    )

            self.tracked_keypoints_pub.publish(
                self.bridge.cv2_to_imgmsg(vis_img, "bgr8", header=left_msg.header)
            )

            # ---- Visualization: feature count (green=tracked, blue=new) ----
            vis_feature_img = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            for pt in primary_new_2d:
                cv2.circle(
                    vis_feature_img,
                    (int(pt.ravel()[0]), int(pt.ravel()[1])),
                    3,
                    (0, 255, 0),
                    -1,
                )
            if new_features_kp is not None:
                for pt in new_features_kp:
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
            self.frame_buffer.clear()
        except Exception as e:
            status_text = f"Error: {e}"
            self.get_logger().error(
                f"Error in processing: {e}\n{traceback.format_exc()}"
            )
            self.prev_gray = None
            self.frame_buffer.clear()

        # ---- Debug window overlay ----
        if vis_img is not None:
            cv2.putText(
                vis_img,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.imshow("VIO Frontend (RAFT)", vis_img)
            cv2.waitKey(1)

    # ==================== Utility: Feature Detection ====================
    def _detect_features(self, image, mask=None):
        """Shi-Tomasi corner detection, returns Nx1x2 array or None."""
        corners = cv2.goodFeaturesToTrack(image, mask=mask, **self.feature_params)
        if corners is not None:
            return corners.reshape(-1, 1, 2)
        return None

    # ==================== Utility: Stereo Triangulation ====================
    def triangulate_stereo(self, img_l, img_r, kp_l, fallback_3d=None):
        """Disparity-based triangulation. Falls back to propagated 3D if stereo fails."""
        if kp_l is None or len(kp_l) == 0:
            return None, np.zeros((0, 3))

        kp_l = kp_l.astype(np.float32)
        # Match left keypoints in right image via LK optical flow (stereo only)
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
                d = pl_flat[0] - pr_flat[0]  # horizontal disparity

                # TUNABLE: min disparity > 1.0 px – rejects points at near-infinity
                if d > 1.0:
                    z = (self.fx * self.baseline) / d
                    x = (pl_flat[0] - self.cx) * z / self.fx
                    y = (pl_flat[1] - self.cy) * z / self.fy
                    points3d.append([x, y, z])
                    good_keypoints.append(pl)
                    valid_point = True

            # Use propagated 3D point if stereo matching failed for this keypoint
            if not valid_point and fallback_3d is not None:
                if not np.isnan(fallback_3d[i, 0]):
                    points3d.append(fallback_3d[i])
                    good_keypoints.append(pl)

        if not good_keypoints:
            return None, np.zeros((0, 3))

        return np.array(good_keypoints, dtype=np.float32).reshape(-1, 1, 2), np.array(
            points3d, dtype=np.float32
        )

    # ==================== Pose Publishing ====================
    def publish_visual_odom(self, rvec, tvec, header, inlier_count):
        """Invert PnP result, accumulate global pose, publish in body frame."""
        # PnP gives prev→current transform; invert to get camera pose
        R_mat, _ = cv2.Rodrigues(rvec)
        R_inv = R_mat.T
        t_inv = -R_inv @ tvec

        # TUNABLE: motion gate – reject frames with unreasonably large jumps
        step_distance = np.linalg.norm(t_inv)
        if step_distance > 1.0:  # metres per frame
            self.get_logger().warn(
                f"Huge jump detected ({step_distance:.2f}m). Ignoring frame."
            )
            return

        # Accumulate incremental transform into global camera pose
        step_transform = np.eye(4)
        step_transform[:3, :3] = R_inv
        step_transform[:3, 3] = t_inv.flatten()
        self.global_transform = self.global_transform @ step_transform

        # Camera frame → body/IMU frame
        global_body = self.T_body_cam @ self.global_transform @ self.T_body_cam_inv
        global_t = global_body[:3, 3]
        global_R = global_body[:3, :3]
        q = R.from_matrix(global_R).as_quat()

        # Build ROS message
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

        # ========== TUNABLE: Dynamic VO Covariance ==========
        # Variance inversely proportional to inlier count.
        # More inliers → lower variance → backend EKF trusts VO more.
        # TUNABLE: 0.1 scaling factor, floor (0.0001), cap (0.5)
        variance_val = 1.0 / (inlier_count + 1) * 0.1
        variance_val = max(variance_val, 0.0001)  # floor
        variance_val = min(variance_val, 0.5)      # cap

        # Position covariance diagonal (x, y, z)
        msg.pose.covariance[0] = variance_val
        msg.pose.covariance[7] = variance_val
        msg.pose.covariance[14] = variance_val
        # Orientation covariance (roll, pitch, yaw) – scaled 2× position
        orientation_var = variance_val * 2.0
        msg.pose.covariance[21] = orientation_var
        msg.pose.covariance[28] = orientation_var
        msg.pose.covariance[35] = orientation_var

        self.vo_pub.publish(msg)


# ========================== Entry Point ==========================
def main(args=None):
    rclpy.init(args=args)
    node = VIOFrontendRAFT()
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
