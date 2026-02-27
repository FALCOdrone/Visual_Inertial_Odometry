import cv2
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")


def _get_safe_device(requested: str) -> str:
    """
    Return 'cuda' only if CUDA is available AND a simple op succeeds.
    Falls back to 'cpu' if the kernel image is incompatible with the GPU.
    """
    if requested != "cuda":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        print(f"⚠️  CUDA unavailable at runtime ({e}). Falling back to CPU.")
        return "cpu"


class FeatureExtractor:
    """Feature extraction and matching using SuperPoint + LightGlue."""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize SuperPoint + LightGlue feature extractor.

        Args:
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = _get_safe_device(device)
        print(f"Using device: {self.device}")

        try:
            from lightglue import LightGlue, SuperPoint

            self.superpoint = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.lightglue = LightGlue(features="superpoint").eval().to(self.device)

            print("✓ SuperPoint + LightGlue models loaded successfully")
        except ImportError:
            print(
                "❌ lightglue not installed. Install with: "
                "pip install git+https://github.com/cvg/LightGlue.git"
            )
            self.superpoint = None
            self.lightglue = None

    def detect_features(self, image):
        """
        Detect keypoints and descriptors using SuperPoint.

        Args:
            image: Grayscale image (numpy array, uint8 [0-255])

        Returns:
            SuperPoint output dict with 'keypoints', 'descriptors', 'scores'
        """
        if self.superpoint is None:
            return None

        # Convert to float tensor, shape (1, 1, H, W), normalized to [0, 1]
        if isinstance(image, np.ndarray):
            data = torch.from_numpy(image).float().to(self.device)
        else:
            data = image.float().to(self.device)

        if data.dim() == 2:  # (H, W)    -> (1, 1, H, W)
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.dim() == 3:  # (C, H, W) -> (1, C, H, W)
            data = data.unsqueeze(0)

        data = data / 255.0  # normalize to [0, 1]

        with torch.no_grad():
            features = self.superpoint({"image": data})

        return features

    def match_features(self, features0, features1):
        """
        Match features between two images using LightGlue.

        Args:
            features0: SuperPoint output dict from first image
            features1: SuperPoint output dict from second image

        Returns:
            Dictionary with match indices and confidence scores
        """
        if self.lightglue is None:
            return None

        with torch.no_grad():
            # LightGlue expects {'image0': feats0, 'image1': feats1}
            result = self.lightglue({"image0": features0, "image1": features1})

        # Extract matches: shape (B, N), -1 means unmatched
        matches_np = result["matches0"][0].cpu().numpy()
        confidence = result["matching_scores0"][0].cpu().numpy()

        # Extract keypoints as numpy arrays (remove batch dim)
        kpts0 = features0["keypoints"][0].cpu().numpy()  # (N, 2)
        kpts1 = features1["keypoints"][0].cpu().numpy()  # (M, 2)

        return {
            "matches": matches_np,
            "confidence": confidence,
            "keypoints0": kpts0,
            "keypoints1": kpts1,
        }

    def visualize_matches(self, left_image, right_image, matches, max_matches=50):
        """
        Visualize feature matches between stereo pair.

        Args:
            left_image: Left camera image
            right_image: Right camera image
            matches: Match results from match_features()
            max_matches: Maximum number of matches to display

        Returns:
            Visualization image
        """
        if matches is None:
            return None

        kpts0 = matches["keypoints0"]
        kpts1 = matches["keypoints1"]
        match_indices = matches["matches"]
        confidence = matches["confidence"]

        # Filter invalid matches (value < 0 means no match)
        valid_matches = match_indices >= 0
        valid_indices = np.where(valid_matches)[0]

        if len(valid_indices) == 0:
            print("No valid matches found")
            return None

        # Sort by confidence and take top matches
        top_k = min(max_matches, len(valid_indices))
        top_match_indices = valid_indices[
            np.argsort(-confidence[valid_indices])[:top_k]
        ]

        # Prepare visualization
        h0, w0 = left_image.shape
        h1, w1 = right_image.shape
        vis = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)

        # Place images side by side
        vis[:h0, :w0] = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        vis[:h1, w0:] = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

        # Draw matches
        for idx in top_match_indices:
            pt0 = kpts0[idx].astype(int)
            pt1 = kpts1[match_indices[idx]].astype(int)

            # Draw circles at keypoints
            cv2.circle(vis, tuple(pt0), 3, (0, 255, 0), -1)
            cv2.circle(vis, (pt1[0] + w0, pt1[1]), 3, (0, 255, 0), -1)

            # Draw lines connecting matches
            cv2.line(vis, tuple(pt0), (pt1[0] + w0, pt1[1]), (0, 255, 255), 1)

        return vis

    def _apply_circular_check(
        self,
        temporal_left,
        stereo_curr,
        temporal_right,
        stereo_prev,
        kpts_l_prev,
        kpts_l_curr,
        kpts_r_curr,
        kpts_r_prev,
        pixel_threshold=2.0,
    ):
        """
        4-way circular check on stereo-temporal tracks.

        Chain: L_{t-1} → L_t → R_t → R_{t-1} → L_{t-1}

        Starting from keypoint i in L_{t-1}, the algorithm follows four
        successive LightGlue matches and measures the pixel reprojection
        error when the loop is closed back to L_{t-1}.  Only tracks whose
        reprojection error is <= pixel_threshold are kept.

        Args:
            temporal_left:  match_features(L_{t-1}, L_t)
            stereo_curr:    match_features(L_t,     R_t)
            temporal_right: match_features(R_t,     R_{t-1})
            stereo_prev:    match_features(L_{t-1}, R_{t-1})
            kpts_*:         (N, 2) numpy keypoint arrays for each image
            pixel_threshold: maximum allowed reprojection error (pixels)

        Returns:
            dict with per-track indices into each image frame, corresponding
            keypoint coordinates, per-track pixel errors, and total count.
        """
        m_ll = temporal_left["matches"]  # shape (N_l_prev,)
        m_lr = stereo_curr["matches"]  # shape (N_l_curr,)
        m_rr = temporal_right["matches"]  # shape (N_r_curr,)
        m_rl = stereo_prev["matches"]  # shape (N_l_prev,)  L_prev→R_prev

        # Build inverse stereo_prev map: R_{t-1}[r] → L_{t-1}[l]
        n_r_prev = kpts_r_prev.shape[0]
        inv_m_rl = np.full(n_r_prev, -1, dtype=np.int32)
        for l_idx, r_idx in enumerate(m_rl):
            if r_idx >= 0:
                inv_m_rl[int(r_idx)] = l_idx

        l_prev_idx, l_curr_idx, r_curr_idx, r_prev_idx, errors = [], [], [], [], []

        for i, j in enumerate(m_ll):
            if j < 0:
                continue
            j = int(j)
            k = int(m_lr[j]) if j < len(m_lr) else -1
            if k < 0:
                continue
            l = int(m_rr[k]) if k < len(m_rr) else -1
            if l < 0 or l >= n_r_prev:
                continue
            closing_i = int(inv_m_rl[l])
            if closing_i < 0:
                continue
            err = float(np.linalg.norm(kpts_l_prev[i] - kpts_l_prev[closing_i]))
            if err <= pixel_threshold:
                l_prev_idx.append(i)
                l_curr_idx.append(j)
                r_curr_idx.append(k)
                r_prev_idx.append(l)
                errors.append(err)

        li = np.array(l_prev_idx, dtype=np.int32)
        lci = np.array(l_curr_idx, dtype=np.int32)
        rci = np.array(r_curr_idx, dtype=np.int32)
        rpi = np.array(r_prev_idx, dtype=np.int32)
        return {
            "l_prev_indices": li,
            "l_curr_indices": lci,
            "r_curr_indices": rci,
            "r_prev_indices": rpi,
            "kpts_l_prev": kpts_l_prev[li] if len(li) else np.empty((0, 2)),
            "kpts_l_curr": kpts_l_curr[lci] if len(lci) else np.empty((0, 2)),
            "kpts_r_curr": kpts_r_curr[rci] if len(rci) else np.empty((0, 2)),
            "kpts_r_prev": kpts_r_prev[rpi] if len(rpi) else np.empty((0, 2)),
            "pixel_errors": np.array(errors, dtype=np.float32),
            "count": len(l_prev_idx),
        }

    def process_full_frame(
        self,
        left_image,
        right_image,
        prev_left_features=None,
        prev_right_features=None,
        pixel_threshold=2.0,
    ):
        """
        Detect features, match the stereo pair, and — when previous features
        are supplied — run temporal matching + 4-way circular check.

        Circular chain:  L_{t-1} → L_t → R_t → R_{t-1} → L_{t-1}

        Args:
            left_image:           Current left grayscale image (numpy uint8)
            right_image:          Current right grayscale image (numpy uint8)
            prev_left_features:   SuperPoint output from previous left frame
                                  (None on the first frame)
            prev_right_features:  SuperPoint output from previous right frame
                                  (None on the first frame)
            pixel_threshold:      Max reprojection error (px) to accept a track

        Returns:
            dict with keys:
              left_features, right_features,
              stereo_matches  (and 'matches' alias for backward compat),
              temporal_matches  – None on first frame, else dict with
                                  'temporal_left', 'temporal_right', 'stereo_prev'
              circular_tracks   – None on first frame, else output of
                                  _apply_circular_check
        """
        feats_l = self.detect_features(left_image)
        feats_r = self.detect_features(right_image)

        if feats_l is None or feats_r is None:
            return None

        stereo = self.match_features(feats_l, feats_r)

        result = {
            "left_features": feats_l,
            "right_features": feats_r,
            "stereo_matches": stereo,
            "matches": stereo,  # backward-compat alias
            "temporal_matches": None,
            "circular_tracks": None,
        }

        if prev_left_features is None or prev_right_features is None:
            return result

        # --- Three additional LightGlue passes for temporal matching ---
        # 1.  L_{t-1} → L_t
        temporal_left = self.match_features(prev_left_features, feats_l)
        # 2.  R_t → R_{t-1}   (forward chain direction)
        temporal_right = self.match_features(feats_r, prev_right_features)
        # 3.  L_{t-1} → R_{t-1}  (needed to close the 4-way loop)
        stereo_prev = self.match_features(prev_left_features, prev_right_features)

        result["temporal_matches"] = {
            "temporal_left": temporal_left,  # L_{t-1} → L_t
            "temporal_right": temporal_right,  # R_t     → R_{t-1}
            "stereo_prev": stereo_prev,  # L_{t-1} → R_{t-1}
        }

        # --- 4-way circular check ---
        kpts_l_prev = prev_left_features["keypoints"][0].cpu().numpy()
        kpts_l_curr = feats_l["keypoints"][0].cpu().numpy()
        kpts_r_curr = feats_r["keypoints"][0].cpu().numpy()
        kpts_r_prev = prev_right_features["keypoints"][0].cpu().numpy()

        result["circular_tracks"] = self._apply_circular_check(
            temporal_left,
            stereo,
            temporal_right,
            stereo_prev,
            kpts_l_prev,
            kpts_l_curr,
            kpts_r_curr,
            kpts_r_prev,
            pixel_threshold=pixel_threshold,
        )

        return result

    def visualize_temporal_tracks(
        self,
        curr_image,
        circular_tracks,
        max_tracks=200,
    ):
        """
        Visualize circular-checked temporal tracks overlaid on the current
        single left (cam0) frame.

        For every accepted track the method draws:
          • A small filled dot at the *previous* keypoint position (ghost)
          • An open circle outline at the *current* keypoint position
          • A line connecting previous → current (motion trail)

        All circular-verified tracks are drawn in green.  An on-screen
        counter shows the total number of active tracks.

        Args:
            curr_image:      Current left grayscale image (numpy uint8)
            circular_tracks: Output of _apply_circular_check / process_full_frame
            max_tracks:      Maximum number of tracks to render

        Returns:
            BGR visualization image (same size as curr_image), or None.
        """
        if circular_tracks is None or circular_tracks["count"] == 0:
            return None

        n = min(max_tracks, circular_tracks["count"])
        kpts_prev = circular_tracks["kpts_l_prev"][:n]
        kpts_curr = circular_tracks["kpts_l_curr"][:n]

        vis = cv2.cvtColor(curr_image, cv2.COLOR_GRAY2BGR)

        track_color = (0, 220, 0)  # green — motion trail
        curr_color = (0, 255, 0)  # bright green — current position outline
        prev_color = (0, 140, 0)  # darker green — previous position ghost

        for idx in range(n):
            pt_prev = tuple(kpts_prev[idx].astype(int))
            pt_curr = tuple(kpts_curr[idx].astype(int))

            # Motion trail line
            cv2.line(vis, pt_prev, pt_curr, track_color, 1, cv2.LINE_AA)

            # Ghost dot at previous position
            cv2.circle(vis, pt_prev, 2, prev_color, -1, cv2.LINE_AA)

            # Open circle outline at current position
            cv2.circle(vis, pt_curr, 4, curr_color, 1, cv2.LINE_AA)

        # On-screen track counter
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
