import cv2
import numpy as np


class FeatureExtractor:
    """Feature tracking using Shi-Tomasi corners + KLT optical flow.

    Temporal correspondences are validated with a forward-backward circular
    consistency check.  Stereo correspondences are found with the same check.


    Features dict schema (stored between frames):
        keypoints   – (1, N, 2) float32, left keypoint positions
        stereo_mask – (N,) bool, True where the stereo KLT check passed
        image       – (H, W) uint8, the grayscale image (needed for KLT)

    The right features dict mirrors this for the right image; its keypoints
    array contains tracked positions for ALL N left points (index-aligned),
    so that kpts_r[i] is the stereo match of kpts_l[i] iff stereo_mask[i].
    """

    def __init__(
        self,
        max_corners: int = 100,       # cap on corners detected per frame; higher → more tracks (prev 500)
                                      #   and better PnP conditioning, but more CPU per frame 
        quality_level: float = 0.5,  # fraction of the strongest corner's score below which  (prev = 0.01)
                                      #   candidates are rejected; lower → more (weaker) corners,
                                      #   higher → fewer but more stable corners
        min_distance: int = 20,       # minimum pixel gap between any two detected corners; (prev 10)
                                      #   larger → more spatially spread-out features (better
                                      #   for geometry), smaller → corners can cluster together
        win_size: tuple = (21, 21),   # KLT patch size searched at each pyramid level; larger (prev 21,21)
                                      #   handles faster motion but is slower and can drift on
                                      #   low-texture regions; smaller is faster but loses tracks
                                      #   under large inter-frame displacement
        max_level: int = 3,           # pyramid depth for KLT; higher levels let the tracker (prev 3)
                                      #   handle larger motions (each level halves the image),
                                      #   0 disables the pyramid (only works for tiny motions)
    ):
        self.max_corners = max_corners
        self.min_distance = min_distance
        self._shi_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7,
        )
        self._lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _detect(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Shi-Tomasi detection.  Returns (N, 2) float32."""
        corners = cv2.goodFeaturesToTrack(image, mask=mask, **self._shi_params)
        return (
            corners.reshape(-1, 2)
            if corners is not None
            else np.empty((0, 2), np.float32)
        )

    def _exclusion_mask(self, shape, pts: np.ndarray, radius: int) -> np.ndarray:
        """255-image with circles of `radius` zeroed around each existing point."""
        mask = np.full(shape[:2], 255, dtype=np.uint8)
        for p in pts.astype(np.int32):
            cv2.circle(mask, tuple(p), radius, 0, -1)
        return mask

    def _track(self, img0: np.ndarray, img1: np.ndarray, pts0: np.ndarray):
        """KLT: track pts0 (N, 2) from img0 to img1.

        Returns:
            pts1    (N, 2) float32 – tracked positions in img1
            status  (N,)   bool   – True where LK succeeded
        """
        if len(pts0) == 0:
            return np.empty((0, 2), np.float32), np.zeros(0, dtype=bool)
        pts1, status, _ = cv2.calcOpticalFlowPyrLK(
            img0,
            img1,
            pts0.reshape(-1, 1, 2).astype(np.float32),
            None,
            **self._lk_params,
        )
        return pts1.reshape(-1, 2), status.squeeze().astype(bool)

    def _circular_track(self, img0, img1, pts0, threshold):
        """Forward-backward circular consistency check.

        Track pts0 forward (img0 → img1) then backward (img1 → img0).
        Accept only points where ||pts0 - back-projected|| < threshold.

        Returns:
            pts1   (N, 2) – tracked positions in img1 (all, including failures)
            mask   (N,)   – bool, True where the track passed the check
            fb_err (N,)   – per-point forward-backward reprojection error
        """
        pts1, s_fwd = self._track(img0, img1, pts0)
        pts_back, s_bwd = self._track(img1, img0, pts1)
        fb_err = np.linalg.norm(pts0 - pts_back, axis=1)
        return pts1, s_fwd & s_bwd & (fb_err < threshold), fb_err

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_features(self, image: np.ndarray) -> dict:
        """Detect Shi-Tomasi corners and return a features dict."""
        pts = self._detect(image)
        n = len(pts)
        return {
            "keypoints": pts.reshape(1, -1, 2),
            "stereo_mask": np.zeros(n, dtype=bool),
            "image": image,
        }

    def process_full_frame(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        prev_left_features: dict = None,
        prev_right_features: dict = None,
        pixel_threshold: float = 2.0,
    ) -> dict:
        """Detect/track features in the current stereo pair.

        First frame (prev_left_features is None):
          • Detects Shi-Tomasi corners on the left image.

        Subsequent frames:
          • Tracks corners from the previous left frame using KLT with a
            forward-backward circular consistency check.
          • Re-detects in uncovered image regions when track count falls below
            half the target, using an exclusion mask to avoid duplicates.
          • Recovers right-image correspondences for the previous frame from
            prev_right_features so that vio_node can triangulate 3-D landmarks.

        In both cases stereo correspondences are found by tracking left → right
        with the same forward-backward circular check.

        Args:
            left_image:          Current left grayscale image (H, W uint8).
            right_image:         Current right grayscale image (H, W uint8).
            prev_left_features:  Features dict from the previous left frame,
                                 or None on the first frame.
            prev_right_features: Features dict from the previous right frame,
                                 or None on the first frame.
            pixel_threshold:     Forward-backward reprojection threshold (px).

        Returns:
            dict with keys:
              left_features   – {"keypoints": (1, N, 2), "stereo_mask": (N,),
                                 "image": ndarray}
              right_features  – {"keypoints": (1, N, 2), "stereo_mask": (N,),
                                 "image": ndarray}  (index-aligned with left)
              stereo_matches  – {"matches": (N,), "keypoints0": (N,2),
                                 "keypoints1": (N,2)}
              matches         – alias for stereo_matches
              temporal_matches – None on first frame, else
                                 {"temporal_left": {"matches": (N_prev,)},
                                  "temporal_right": None, "stereo_prev": None}
              circular_tracks  – None on first frame, else dict with:
                                   count, kpts_l_prev, kpts_l_curr,
                                   kpts_r_prev (for triangulation),
                                   kpts_r_curr, pixel_errors
        """
        temporal_matches = None
        circular_tracks = None

        if prev_left_features is None:
            # ── First frame: fresh Shi-Tomasi detection ───────────────────────
            pts_l_curr = self._detect(left_image)
        else:
            # ── Subsequent frames: temporal KLT + circular check ──────────────
            pts_l_prev = prev_left_features["keypoints"].reshape(-1, 2)
            prev_img = prev_left_features["image"]

            pts_l_fwd, mask_circ, fb_errors = self._circular_track(
                prev_img, left_image, pts_l_prev, pixel_threshold
            )

            good_prev_idx = np.where(mask_circ)[0].astype(np.int32)
            good_pts_l_prev = pts_l_prev[mask_circ]
            good_pts_l_curr = pts_l_fwd[mask_circ]

            pts_l_curr = good_pts_l_curr

            # Supplement with fresh detections if below half the target count
            if len(pts_l_curr) < self.max_corners // 2:
                excl = (
                    self._exclusion_mask(
                        left_image.shape, pts_l_curr, self.min_distance
                    )
                    if len(pts_l_curr) > 0
                    else None
                )
                new_pts = self._detect(left_image, mask=excl)
                need = self.max_corners - len(pts_l_curr)
                if len(new_pts) > 0:
                    pts_l_curr = (
                        np.vstack([pts_l_curr, new_pts[:need]])
                        if len(pts_l_curr) > 0
                        else new_pts[:need]
                    )

            # temporal_matches: prev-keypoint index → curr-keypoint index, -1 if lost
            n_prev = len(pts_l_prev)
            tl_matches = np.full(n_prev, -1, dtype=np.int32)
            for new_idx, prev_idx in enumerate(good_prev_idx):
                tl_matches[prev_idx] = new_idx
            temporal_matches = {
                "temporal_left": {"matches": tl_matches},
                "temporal_right": None,
                "stereo_prev": None,
            }

            # Recover right-frame correspondences for the previous frame so
            # that vio_node can triangulate 3-D landmarks (kpts_l_prev + kpts_r_prev).
            prev_stereo_valid = prev_left_features.get("stereo_mask", None)
            prev_right_pts = (
                prev_right_features["keypoints"].reshape(-1, 2)
                if prev_right_features is not None
                else None
            )

            if prev_stereo_valid is not None and prev_right_pts is not None:
                has_stereo = prev_stereo_valid[good_prev_idx]
                tri_sel = np.where(has_stereo)[0]  # into good_prev_idx
                kpts_l_prev_tri = good_pts_l_prev[tri_sel]
                kpts_l_curr_tri = good_pts_l_curr[tri_sel]
                kpts_r_prev_tri = prev_right_pts[good_prev_idx[tri_sel]]
                fb_errors_tri = fb_errors[mask_circ][tri_sel]
                tri_count = len(tri_sel)
            else:
                kpts_l_prev_tri = good_pts_l_prev
                kpts_l_curr_tri = good_pts_l_curr
                kpts_r_prev_tri = np.empty((0, 2), np.float32)
                fb_errors_tri = fb_errors[mask_circ]
                tri_count = len(good_pts_l_prev)

            circular_tracks = {
                "count": tri_count,
                "kpts_l_prev": kpts_l_prev_tri,
                "kpts_l_curr": kpts_l_curr_tri,
                "kpts_r_prev": kpts_r_prev_tri,
                "kpts_r_curr": np.empty((0, 2), np.float32),
                "l_prev_indices": good_prev_idx,
                "l_curr_indices": np.arange(len(good_pts_l_curr), dtype=np.int32),
                "r_curr_indices": np.empty(0, np.int32),
                "r_prev_indices": np.empty(0, np.int32),
                "pixel_errors": fb_errors_tri.astype(np.float32),
            }

        # ── Stereo: track left → right with circular check ────────────────────
        if len(pts_l_curr) > 0:
            pts_r_curr, mask_stereo, _ = self._circular_track(
                left_image, right_image, pts_l_curr, pixel_threshold
            )
        else:
            pts_r_curr = np.empty((0, 2), np.float32)
            mask_stereo = np.zeros(0, dtype=bool)

        n = len(pts_l_curr)
        # matches[i] = i when track i succeeded (1-to-1 correspondence), else -1
        stereo_match_idx = np.where(mask_stereo, np.arange(n), -1).astype(np.int32)

        # ── Pack features dicts (keypoints shape: (1, N, 2)) ─────────────────
        def _pack(pts, img, smask):
            kp = (
                pts.reshape(1, -1, 2)
                if len(pts) > 0
                else np.empty((1, 0, 2), np.float32)
            )
            return {"keypoints": kp, "stereo_mask": smask, "image": img}

        stereo_matches = {
            "matches": stereo_match_idx,
            "keypoints0": (
                pts_l_curr if len(pts_l_curr) > 0 else np.empty((0, 2), np.float32)
            ),
            "keypoints1": (
                pts_r_curr if len(pts_r_curr) > 0 else np.empty((0, 2), np.float32)
            ),
        }

        return {
            "left_features": _pack(pts_l_curr, left_image, mask_stereo),
            "right_features": _pack(pts_r_curr, right_image, mask_stereo),
            "stereo_matches": stereo_matches,
            "matches": stereo_matches,
            "temporal_matches": temporal_matches,
            "circular_tracks": circular_tracks,
        }

    def visualize_matches(self, left_image, right_image, matches, max_matches=50):
        """Stereo match visualization (left | right side-by-side)."""
        if matches is None:
            return None

        kpts0 = matches["keypoints0"]
        kpts1 = matches["keypoints1"]
        match_idx = matches["matches"]

        valid = np.where(match_idx >= 0)[0]
        if len(valid) == 0:
            return None

        top_k = valid[:max_matches]

        h0, w0 = left_image.shape[:2]
        h1, w1 = right_image.shape[:2]
        vis = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
        vis[:h0, :w0] = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        vis[:h1, w0:] = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

        for idx in top_k:
            pt0 = tuple(kpts0[idx].astype(int))
            pt1_r = kpts1[match_idx[idx]].astype(int)
            pt1 = (int(pt1_r[0] + w0), int(pt1_r[1]))
            cv2.circle(vis, pt0, 3, (0, 255, 0), -1)
            cv2.circle(vis, pt1, 3, (0, 255, 0), -1)
            cv2.line(vis, pt0, pt1, (0, 255, 255), 1)

        return vis

    def visualize_temporal_tracks(self, curr_image, circular_tracks, max_tracks=200):
        """Temporal track visualization overlaid on the current left frame.

        Draws motion trails (prev → curr) for all circular-checked tracks.
        """
        if circular_tracks is None or circular_tracks["count"] == 0:
            return None

        n = min(max_tracks, circular_tracks["count"])
        kpts_prev = circular_tracks["kpts_l_prev"][:n]
        kpts_curr = circular_tracks["kpts_l_curr"][:n]

        vis = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)

        for i in range(n):
            pt_prev = tuple(kpts_prev[i].astype(int))
            pt_curr = tuple(kpts_curr[i].astype(int))
            cv2.line(vis, pt_prev, pt_curr, (0, 220, 0), 1, cv2.LINE_AA)
            cv2.circle(vis, pt_prev, 2, (0, 140, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, pt_curr, 4, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(
            vis,
            f"Tracks: {circular_tracks['count']}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return vis
