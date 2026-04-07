#!/usr/bin/env python3
"""
loop_closure_node.py
====================
Pure-Python / NumPy loop closure for the VIO pipeline.

Architecture
------------
This node sits alongside the ESKF.  It subscribes to the fused ESKF odometry
(for drift-free pose initialization) and to rectified left-camera images
(already published by stereo_rectifier_node on /cam0/image_rect).

At VIO-keyframe rate (~5 Hz) it:
  1. Extracts ORB descriptors from the rectified image.
  2. Builds / queries an online TF-IDF bag-of-words (BoW) database to score
     similarity against all previous keyframes.
  3. When a high-scoring candidate is found AND geometric verification passes
     (Essential-matrix RANSAC with ≥ min_lc_inliers inliers), it accepts the
     loop closure.
  4. Runs a Gauss-Newton pose-graph optimizer on all keyframe poses (odometry
     edges + the new loop edge) to distribute the correction.
  5. Computes the correction transform Δ = T_new_latest * T_old_latest^{-1}
     and publishes it on /lc/correction (geometry_msgs/PoseStamped) so the
     ESKF can reinitialise.
  6. Publishes the globally-corrected trajectory on /lc/global_trajectory
     (nav_msgs/Path) for visualization.

BoW implementation
------------------
We use an online k-means visual vocabulary over ORB descriptors:
  - Vocabulary is built after the first `vocab_min_keyframes` keyframes using
    k-means with `vocab_size` cluster centers.
  - Each keyframe is represented as a TF-IDF histogram over the vocabulary.
  - Loop score = cosine similarity between TF-IDF histograms.

Pose graph
----------
Node variables: T_i ∈ SE(3), parameterized as (p_i ∈ R^3, phi_i ∈ R^3)
  where R_i = exp_so3(phi_i) around the linearization point.

Edges:
  - Consecutive odometry edges (from ESKF poses between keyframes)
  - Loop closure edges (from geometric verification)

Gauss-Newton convergence is fast (~3 iterations) because the graph is
relatively sparse and the initialization from the ESKF is already good.

Subscriptions
-------------
  /eskf/odometry   nav_msgs/Odometry      fused ESKF pose @ ~100-200 Hz
  /cam0/image_rect sensor_msgs/Image      rectified left image @ 20 Hz

Publications
------------
  /lc/correction          geometry_msgs/PoseStamped  SE(3) correction to apply to ESKF
  /lc/corrected_odometry  nav_msgs/Odometry          ESKF pose with LC correction applied
  /lc/global_trajectory   nav_msgs/Path              full corrected global trajectory

Parameters
----------
  All under the `loop_closure` section of pipeline_params.yaml:
  min_keyframe_separation  int   (default 20)   — min KF gap to accept LC candidate
  min_loop_score           float (default 0.015) — TF-IDF cosine sim threshold
  min_lc_inliers           int   (default 20)    — Essential-matrix RANSAC inliers
  max_lc_correction_m      float (default 5.0)   — reject loops needing > N m shift
  lc_pos_std               float (default 0.3)   — LC position noise [m]
  lc_ang_std               float (default 0.05)  — LC angular noise [rad]
  vocab_size               int   (default 256)   — BoW vocabulary cluster count
  vocab_min_keyframes      int   (default 20)    — KFs before building vocabulary
  orb_n_features           int   (default 500)   — ORB features per frame
  pgo_max_iter             int   (default 5)     — Gauss-Newton pose graph iterations
  kf_translation_thresh    float (default 0.05)  — min ESKF motion to trigger a new KF [m]
  kf_rotation_thresh_deg   float (default 3.0)   — min ESKF rotation to trigger a new KF [deg]
  kf_max_age_sec           float (default 0.5)   — max time between keyframes [s]
"""

from __future__ import annotations

import time
import yaml
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

# Reuse SO(3) helpers from the shared utility module
from vio_pipeline.so3_utils import (
    exp_so3,
    log_so3,
    quat_to_rot,
    rot_to_quat,
    quat_mul,
    exp_so3_quat,
)


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class Keyframe:
    """One entry in the keyframe database."""
    kf_id: int
    stamp: float                         # ROS time in seconds (float)
    p: np.ndarray                        # position (3,) in world frame
    R: np.ndarray                        # rotation matrix (3×3) body→world
    descriptors: np.ndarray              # ORB descriptors (N×32) uint8
    keypoints: List[cv2.KeyPoint]        # corresponding keypoints
    bow_vec: Optional[np.ndarray] = None  # TF-IDF histogram (vocab_size,)
    pts3d: Optional[np.ndarray] = None   # (N,3) 3-D positions in rect-cam0 frame; NaN = no depth


@dataclass
class PoseGraphEdge:
    """An edge in the pose graph between keyframes i and j."""
    i: int           # index into self._keyframes
    j: int           # index into self._keyframes
    T_ij: np.ndarray  # 4×4 relative transform T_i^{-1} @ T_j
    sqrt_info: np.ndarray  # 6×6 sqrt information matrix (for weighting)
    is_loop: bool = False


# ── BoW vocabulary ─────────────────────────────────────────────────────────────


class OnlineBoW:
    """
    Lightweight online Bag-of-Words using k-means on binary (ORB) descriptors.

    ORB descriptors are uint8 bitmaps.  We work in float space for k-means but
    use Hamming-distance assignment (via popcount on XOR) for scoring.

    TF-IDF weighting:
      TF  = term frequency = count of word w in document d / total words in d
      IDF = inverse document frequency = log(N / df_w + 1)
             where N = total documents, df_w = documents containing word w
      score_w = TF_w * IDF_w
    """

    def __init__(self, vocab_size: int = 256) -> None:
        self._vocab_size = vocab_size
        self._centers: Optional[np.ndarray] = None   # (vocab_size, 32) uint8
        self._idf: np.ndarray = np.ones(vocab_size, dtype=np.float64)
        self._df: np.ndarray = np.zeros(vocab_size, dtype=np.float64)  # doc freq
        self._n_docs: int = 0
        self._ready: bool = False

    @property
    def ready(self) -> bool:
        return self._ready

    def build(self, all_descriptors: np.ndarray) -> None:
        """
        Run k-means on a pool of descriptors to create the vocabulary.

        Parameters
        ----------
        all_descriptors : (N, 32) uint8 array of ORB descriptors.
        """
        if len(all_descriptors) < self._vocab_size:
            return  # not enough data yet

        # Convert to float32 for cv2.kmeans
        data = all_descriptors.astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,   # max iterations — keep small so startup is fast
            1.0,  # epsilon
        )
        # Use KMEANS_PP_CENTERS for better initialization
        _, labels, centers = cv2.kmeans(
            data,
            self._vocab_size,
            None,
            criteria,
            attempts=3,
            flags=cv2.KMEANS_PP_CENTERS,
        )
        self._centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)
        self._ready = True

    def _assign(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Assign each descriptor in (N, 32) to its nearest visual word.

        Returns word indices (N,) using Hamming distance via XOR + popcount.
        Implemented efficiently with broadcasting.
        """
        # XOR each descriptor against every center → Hamming distances (N, vocab_size)
        # np.unpackbits then sum works but is slow for large N.  Use byte popcount trick.
        # descriptors: (N, 32) uint8, centers: (V, 32) uint8
        N = len(descriptors)
        V = self._vocab_size
        # Broadcast XOR: (N, 1, 32) ^ (1, V, 32) → (N, V, 32)
        xor = descriptors[:, None, :].astype(np.uint8) ^ self._centers[None, :, :].astype(np.uint8)
        # Hamming = sum of popcount per byte.  Use np.unpackbits per row.
        # Reshape to (N*V, 32), unpack bits, sum.
        flat_xor = xor.reshape(N * V, 32)
        bits = np.unpackbits(flat_xor, axis=1)   # (N*V, 256)
        ham = bits.sum(axis=1).reshape(N, V)     # (N, V)
        return np.argmin(ham, axis=1)             # (N,)

    def compute_bow(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Compute TF-IDF histogram for a set of descriptors.

        Returns a unit-norm vector of shape (vocab_size,).
        Returns zeros if vocabulary is not ready.
        """
        if not self._ready or len(descriptors) == 0:
            return np.zeros(self._vocab_size, dtype=np.float64)

        words = self._assign(descriptors)
        # TF: normalized word counts
        tf = np.bincount(words, minlength=self._vocab_size).astype(np.float64)
        tf /= max(tf.sum(), 1.0)
        # TF-IDF
        vec = tf * self._idf
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        return vec

    def add_document(self, bow_vec: np.ndarray) -> None:
        """
        Update IDF counts after adding a new document with the given BoW vector.

        Call this AFTER computing bow_vec so the IDF is updated for subsequent
        queries (the current document does not bias its own similarity score).
        """
        self._n_docs += 1
        # Increment document frequency for all non-zero words
        self._df[bow_vec > 0] += 1.0
        # Recompute IDF: log((N+1) / (df_w + 1)) + 1  (smoothed)
        self._idf = np.log(
            (self._n_docs + 1.0) / (self._df + 1.0)
        ) + 1.0

    def score(self, q: np.ndarray, d: np.ndarray) -> float:
        """Cosine similarity between two unit-norm TF-IDF vectors."""
        return float(np.dot(q, d))


# ── Pose graph optimizer ───────────────────────────────────────────────────────


class PoseGraphOptimizer:
    """
    Minimal Gauss-Newton pose graph optimizer on SE(3).

    State: x = [phi_0, p_0, phi_1, p_1, ..., phi_{n-1}, p_{n-1}]
           where T_i = (exp_so3(phi_i), p_i) around the linearization point.

    Each edge (i, j, T_ij, sqrt_info) contributes a 6D residual:
        r_rot = log_so3(R_j^T @ R_i @ R_ij)          ← rotation error
        r_pos = R_i^T @ (p_j - p_i) - t_ij           ← position error (body frame)

    The first pose is fixed (prior) so the system is well-constrained.
    """

    def __init__(self, max_iter: int = 5, damping: float = 1e-6) -> None:
        self._max_iter = max_iter
        self._damping = damping

    def optimize(
        self,
        poses: List[Tuple[np.ndarray, np.ndarray]],  # [(p_i, R_i), ...]
        edges: List[PoseGraphEdge],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Run Gauss-Newton optimization.

        Parameters
        ----------
        poses : list of (p, R) tuples — initial estimates.
        edges : list of PoseGraphEdge.

        Returns
        -------
        Optimized list of (p, R) tuples in the same order.
        """
        n = len(poses)
        if n == 0:
            return poses

        # Work on copies
        ps = [p.copy() for (p, _) in poses]
        Rs = [R.copy() for (_, R) in poses]

        for _iter in range(self._max_iter):
            # Build H (6n×6n) and b (6n,) — fix pose 0 (remove its DOFs)
            H = np.zeros((6 * n, 6 * n), dtype=np.float64)
            b = np.zeros(6 * n, dtype=np.float64)

            for edge in edges:
                i, j = edge.i, edge.j
                if i < 0 or j < 0 or i >= n or j >= n:
                    continue

                R_ij = edge.T_ij[:3, :3]
                t_ij = edge.T_ij[:3, 3]
                Ri, pi = Rs[i], ps[i]
                Rj, pj = Rs[j], ps[j]

                # Rotation residual: r_R = log(R_j^T @ Ri @ R_ij)
                R_err = Rj.T @ Ri @ R_ij
                r_R = log_so3(R_err)           # (3,)

                # Position residual in frame i: r_p = Ri^T @ (pj - pi) - t_ij
                r_p = Ri.T @ (pj - pi) - t_ij  # (3,)

                r = np.concatenate([r_R, r_p])  # (6,)

                # Jacobians w.r.t. pose i (phi_i, p_i) and pose j (phi_j, p_j)
                # ∂r_R/∂phi_i = -J_r^{-1}(r_R) @ Rj^T @ Ri  (approx -I for small r_R)
                # ∂r_R/∂phi_j = +J_r^{-1}(r_R)  (approx +I)
                # ∂r_p/∂phi_i = skew(Ri^T @ (pj - pi))  (first-order)
                # ∂r_p/∂p_i   = -Ri^T
                # ∂r_p/∂p_j   = +Ri^T

                # Use simplified (first-order) Jacobians for efficiency
                Ji = np.zeros((6, 6), dtype=np.float64)
                Jj = np.zeros((6, 6), dtype=np.float64)

                dp = pj - pi
                # rotation block w.r.t. phi_i
                Ji[0:3, 0:3] = -np.eye(3)
                # rotation block w.r.t. phi_j
                Jj[0:3, 0:3] = np.eye(3)
                # position block w.r.t. phi_i: skew(Ri^T @ dp)
                Ji[3:6, 0:3] = _skew(Ri.T @ dp)
                # position block w.r.t. p_i
                Ji[3:6, 3:6] = -Ri.T
                # position block w.r.t. p_j
                Jj[3:6, 3:6] = Ri.T

                # Weight
                W = edge.sqrt_info.T @ edge.sqrt_info   # 6×6 information matrix

                si, sj = 6 * i, 6 * j
                H[si:si+6, si:si+6] += Ji.T @ W @ Ji
                H[si:si+6, sj:sj+6] += Ji.T @ W @ Jj
                H[sj:sj+6, si:si+6] += Jj.T @ W @ Ji
                H[sj:sj+6, sj:sj+6] += Jj.T @ W @ Jj
                b[si:si+6] += Ji.T @ W @ r
                b[sj:sj+6] += Jj.T @ W @ r

            # Fix first pose by zeroing its rows/columns (Dirichlet boundary)
            H[0:6, :] = 0.0
            H[:, 0:6] = 0.0
            H[0:6, 0:6] = np.eye(6)
            b[0:6] = 0.0

            # Tikhonov damping for numerical stability
            H += self._damping * np.eye(6 * n)

            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                break

            # Apply increments
            for k in range(1, n):
                sk = 6 * k
                dphi = dx[sk:sk+3]
                dp_k = dx[sk+3:sk+6]
                Rs[k] = exp_so3(dphi) @ Rs[k]
                ps[k] = ps[k] + dp_k

            if np.linalg.norm(dx) < 1e-6:
                break

        return list(zip(ps, Rs))


def _skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric matrix (module-level for use in PGO)."""
    return np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


# ── ROS2 node ──────────────────────────────────────────────────────────────────


class LoopClosureNode(Node):
    """
    Loop closure node — pure Python / NumPy / OpenCV implementation.

    See module docstring for full architecture description.
    """

    def __init__(self) -> None:
        super().__init__("loop_closure_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("min_keyframe_separation", 20)
        self.declare_parameter("min_loop_score",          0.015)
        self.declare_parameter("min_lc_inliers",          20)
        self.declare_parameter("max_lc_correction_m",     5.0)
        self.declare_parameter("lc_pos_std",              0.3)
        self.declare_parameter("lc_ang_std",              0.05)
        self.declare_parameter("vocab_size",              256)
        self.declare_parameter("vocab_min_keyframes",     20)
        self.declare_parameter("orb_n_features",          500)
        self.declare_parameter("pgo_max_iter",            5)
        self.declare_parameter("kf_translation_thresh",   0.05)
        self.declare_parameter("kf_rotation_thresh_deg",  3.0)
        self.declare_parameter("kf_max_age_sec",          0.5)
        self.declare_parameter("config_path", "")

        self._min_kf_sep     = int(self.get_parameter("min_keyframe_separation").value)
        self._min_score      = float(self.get_parameter("min_loop_score").value)
        self._min_inliers    = int(self.get_parameter("min_lc_inliers").value)
        self._max_corr_m     = float(self.get_parameter("max_lc_correction_m").value)
        self._lc_pos_std     = float(self.get_parameter("lc_pos_std").value)
        self._lc_ang_std     = float(self.get_parameter("lc_ang_std").value)
        vocab_size           = int(self.get_parameter("vocab_size").value)
        self._vocab_min_kf   = int(self.get_parameter("vocab_min_keyframes").value)
        orb_n                = int(self.get_parameter("orb_n_features").value)
        pgo_iter             = int(self.get_parameter("pgo_max_iter").value)
        self._kf_trans_thr   = float(self.get_parameter("kf_translation_thresh").value)
        self._kf_rot_thr_rad = float(self.get_parameter("kf_rotation_thresh_deg").value) * np.pi / 180.0
        self._kf_max_age     = float(self.get_parameter("kf_max_age_sec").value)

        # Load camera extrinsics and stereo params from config file
        config_path = self.get_parameter("config_path").value
        self._R_BS: Optional[np.ndarray] = None          # cam0→body rotation (3×3)
        self._bf: float = 0.0                            # baseline × fx [m·px]
        self._fx_rect: float = 0.0
        self._cx_rect: float = 0.0
        self._cy_rect: float = 0.0
        if config_path:
            self._load_stereo_config(config_path)
        else:
            self.get_logger().warn(
                "LC: no config_path provided — stereo PnP disabled, falling back to E-matrix only"
            )

        self.get_logger().info(
            f"LoopClosureNode params: min_kf_sep={self._min_kf_sep}  "
            f"min_score={self._min_score}  min_inliers={self._min_inliers}  "
            f"max_corr={self._max_corr_m} m  vocab_size={vocab_size}  "
            f"orb_features={orb_n}  pgo_iter={pgo_iter}"
        )

        # ── Internal state ───────────────────────────────────────────────────
        self._orb = cv2.ORB_create(nfeatures=orb_n)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._bow = OnlineBoW(vocab_size=vocab_size)
        self._pgo = PoseGraphOptimizer(max_iter=pgo_iter)

        # Keyframe database — list of Keyframe objects (all-time)
        self._keyframes: List[Keyframe] = []
        # Pose graph edges (odometry + loop closure)
        self._pg_edges: List[PoseGraphEdge] = []

        # Accumulator for vocabulary building
        self._desc_pool: List[np.ndarray] = []
        self._vocab_built = False

        # Latest ESKF pose
        self._latest_p: Optional[np.ndarray] = None
        self._latest_R: Optional[np.ndarray] = None
        self._latest_stamp: float = 0.0

        # Keyframe selection state
        self._last_kf_p: Optional[np.ndarray] = None
        self._last_kf_R: Optional[np.ndarray] = None
        self._last_kf_stamp: float = 0.0

        # LC information matrix
        pos_info = 1.0 / (self._lc_pos_std ** 2)
        ang_info = 1.0 / (self._lc_ang_std ** 2)
        self._lc_sqrt_info = np.diag(
            np.concatenate([
                np.full(3, np.sqrt(ang_info)),
                np.full(3, np.sqrt(pos_info)),
            ])
        )

        # Odometry edge information matrix (tight — derived from VIO noise)
        # Matches typical ESKF position uncertainty between keyframes
        odo_pos_info = 1.0 / (0.05 ** 2)
        odo_ang_info = 1.0 / (0.05 ** 2)
        self._odo_sqrt_info = np.diag(
            np.concatenate([
                np.full(3, np.sqrt(odo_ang_info)),
                np.full(3, np.sqrt(odo_pos_info)),
            ])
        )

        # Loop closure counter
        self._lc_count = 0

        # Rectified camera intrinsics — populated from /cam0/camera_info + /cam1/camera_info
        # via cv2.stereoRectify (same as vio_node). Until received, verification is blocked.
        self._K_rect: Optional[np.ndarray] = None
        self._cam0_info: Optional[CameraInfo] = None
        self._cam1_info: Optional[CameraInfo] = None

        # ── QoS ─────────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self._pub_correction = self.create_publisher(
            PoseStamped, "/lc/correction", 10
        )
        self._pub_odom = self.create_publisher(
            Odometry, "/lc/corrected_odometry", qos_be
        )
        self._pub_path = self.create_publisher(
            Path, "/lc/global_trajectory", qos_be
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            Odometry, "/eskf/odometry", self._eskf_cb, qos_be
        )
        # Synchronized stereo pair for metric 3-D points (PnP loop closure)
        _cam0_sub = Subscriber(self, Image, "/cam0/image_rect", qos_profile=qos_be)
        _cam1_sub = Subscriber(self, Image, "/cam1/image_rect", qos_profile=qos_be)
        self._stereo_sync = ApproximateTimeSynchronizer(
            [_cam0_sub, _cam1_sub], queue_size=10, slop=0.05
        )
        self._stereo_sync.registerCallback(self._stereo_cb)
        self.create_subscription(
            CameraInfo, "/cam0/camera_info", self._cam0_info_cb, 10
        )
        self.create_subscription(
            CameraInfo, "/cam1/camera_info", self._cam1_info_cb, 10
        )

        self.get_logger().info(
            "LoopClosureNode ready — subscribed to /eskf/odometry and stereo /cam0+cam1/image_rect"
        )

    # ── ESKF odometry callback ─────────────────────────────────────────────────

    def _eskf_cb(self, msg: Odometry) -> None:
        """Cache the latest ESKF pose for use when a new image arrives."""
        self._latest_p = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=np.float64)
        q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=np.float64)
        q /= np.linalg.norm(q)
        self._latest_R = quat_to_rot(q)
        self._latest_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    # ── Camera info callbacks (compute rectified K) ────────────────────────────

    def _cam0_info_cb(self, msg: CameraInfo) -> None:
        if self._K_rect is not None:
            return
        self._cam0_info = msg
        self._try_build_K_rect()

    def _cam1_info_cb(self, msg: CameraInfo) -> None:
        if self._K_rect is not None:
            return
        self._cam1_info = msg
        self._try_build_K_rect()

    def _load_stereo_config(self, config_path: str) -> None:
        """Load camera extrinsics and stereo params from the dataset config YAML."""
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception as exc:
            self.get_logger().warn(f"LC: failed to load config '{config_path}': {exc}")
            return

        T_b_c0 = np.array(cfg["cam0"]["T_BS"], dtype=np.float64).reshape(4, 4)
        T_b_c1 = np.array(cfg["cam1"]["T_BS"], dtype=np.float64).reshape(4, 4)

        # R_BS: rotation part of T_BS for cam0 (maps cam0 frame vectors → body frame)
        self._R_BS = T_b_c0[:3, :3].copy()

        # Stereo geometry for disparity → depth
        K0  = np.diag([*cfg["cam0"]["intrinsics"][:2], 1.0])
        K0[0, 2] = cfg["cam0"]["intrinsics"][2]
        K0[1, 2] = cfg["cam0"]["intrinsics"][3]
        D0  = np.array(cfg["cam0"]["distortion"], dtype=np.float64)
        K1  = np.diag([*cfg["cam1"]["intrinsics"][:2], 1.0])
        K1[0, 2] = cfg["cam1"]["intrinsics"][2]
        K1[1, 2] = cfg["cam1"]["intrinsics"][3]
        D1  = np.array(cfg["cam1"]["distortion"], dtype=np.float64)

        T_c1_c0 = np.linalg.inv(T_b_c1) @ T_b_c0
        R_rel   = T_c1_c0[:3, :3]
        t_rel   = T_c1_c0[:3, 3]

        w, h = cfg["cam0"]["resolution"]
        _, _, P0, P1, _, _, _ = cv2.stereoRectify(
            K0, D0, K1, D1, (w, h), R_rel, t_rel, alpha=0
        )
        # Store rectified intrinsics (may be overridden later by camera_info callback)
        self._K_rect    = P0[:3, :3].copy()
        self._fx_rect   = float(P0[0, 0])
        self._cx_rect   = float(P0[0, 2])
        self._cy_rect   = float(P0[1, 2])
        # baseline × fx from the right projection matrix: P1[0,3] = -fx * baseline
        self._bf        = abs(float(P1[0, 3]))

        # Stereo semi-global block matching
        self._stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=7,
            P1=8  * 1 * 7 * 7,
            P2=32 * 1 * 7 * 7,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.get_logger().info(
            f"LC: stereo config loaded — baseline={self._bf / self._fx_rect * 1000:.1f} mm  "
            f"R_BS[0]={np.round(self._R_BS[0], 3)}"
        )

    def _try_build_K_rect(self) -> None:
        """Compute rectified K from both camera_info messages via cv2.stereoRectify."""
        if self._cam0_info is None or self._cam1_info is None:
            return
        m0, m1 = self._cam0_info, self._cam1_info
        h, w = m0.height, m0.width

        K0 = np.array(m0.k, dtype=np.float64).reshape(3, 3)
        D0 = np.array(m0.d, dtype=np.float64)
        K1 = np.array(m1.k, dtype=np.float64).reshape(3, 3)
        D1 = np.array(m1.d, dtype=np.float64)
        # R and P from camera_info are the rectification rotation and projection matrix.
        # If already set by the bag, use them; otherwise run stereoRectify.
        R1_ci = np.array(m0.r, dtype=np.float64).reshape(3, 3)
        P0_ci = np.array(m0.p, dtype=np.float64).reshape(3, 4)
        if not np.allclose(R1_ci, np.eye(3)) and not np.allclose(P0_ci, np.zeros((3, 4))):
            # camera_info already has rectification params (published by stereo_rectifier_node
            # or the bag); extract K_rect directly from P0
            self._K_rect = P0_ci[:3, :3].copy()
        else:
            # Fallback: run stereoRectify ourselves with a rough baseline estimate
            # T from cam0 to cam1 in cam0 frame (EuRoC: ~0.11 m along x)
            T_c0_c1 = np.array([-0.11, 0.0, 0.0], dtype=np.float64)
            R_c0_c1 = np.eye(3, dtype=np.float64)
            _, _, _, _, P0_rect, _, _ = cv2.stereoRectify(
                K0, D0, K1, D1, (w, h), R_c0_c1, T_c0_c1, alpha=0
            )
            self._K_rect = P0_rect[:3, :3].copy()

        # Keep back-projection params in sync with K_rect
        self._fx_rect = float(self._K_rect[0, 0])
        self._cx_rect = float(self._K_rect[0, 2])
        self._cy_rect = float(self._K_rect[1, 2])

        self.get_logger().info(
            f"LC: rectified K built: fx={self._K_rect[0,0]:.1f} fy={self._K_rect[1,1]:.1f} "
            f"cx={self._K_rect[0,2]:.1f} cy={self._K_rect[1,2]:.1f}"
        )

    # ── Stereo image callback ───────────────────────────────────────────────────

    def _stereo_cb(self, msg0: Image, msg1: Image) -> None:
        """
        Synchronized stereo callback.  Runs at camera rate (20 Hz) but only
        does heavy work on keyframes (~5 Hz).
        """
        if self._latest_p is None:
            return  # ESKF not yet initialized

        stamp = msg0.header.stamp.sec + msg0.header.stamp.nanosec * 1e-9

        if not self._should_be_keyframe(stamp):
            return

        # Decode both images to grayscale
        try:
            img0 = self._decode_image(msg0)
            img1 = self._decode_image(msg1)
        except Exception as exc:
            self.get_logger().warn(f"LC: image decode failed: {exc}")
            return

        # Extract ORB descriptors on cam0
        kps, descs = self._orb.detectAndCompute(img0, None)
        if descs is None or len(descs) < 10:
            self.get_logger().debug("LC: too few ORB features, skipping keyframe")
            return

        # Compute stereo 3-D points for each keypoint (NaN if no valid depth)
        pts3d = self._compute_pts3d(img0, img1, kps)

        # Snapshot pose at keyframe time
        p_kf = self._latest_p.copy()
        R_kf = self._latest_R.copy()

        # Create keyframe
        kf_id = len(self._keyframes)
        kf = Keyframe(
            kf_id=kf_id,
            stamp=stamp,
            p=p_kf,
            R=R_kf,
            descriptors=descs,
            keypoints=kps,
            pts3d=pts3d,
        )

        # Update keyframe selection state
        self._last_kf_p = p_kf
        self._last_kf_R = R_kf
        self._last_kf_stamp = stamp

        # Accumulate descriptors for vocabulary building
        self._desc_pool.append(descs)

        # Build vocabulary once we have enough keyframes
        if not self._vocab_built and kf_id >= self._vocab_min_kf - 1:
            self._build_vocabulary()

        # Compute BoW vector (zero until vocab is ready)
        if self._bow.ready:
            kf.bow_vec = self._bow.compute_bow(descs)

        # Add odometry edge from previous keyframe
        if len(self._keyframes) > 0:
            prev_kf = self._keyframes[-1]
            T_ij = self._relative_pose(prev_kf.p, prev_kf.R, p_kf, R_kf)
            edge = PoseGraphEdge(
                i=len(self._keyframes) - 1,
                j=kf_id,
                T_ij=T_ij,
                sqrt_info=self._odo_sqrt_info.copy(),
                is_loop=False,
            )
            self._pg_edges.append(edge)

        # Store keyframe
        self._keyframes.append(kf)

        # Update BoW database IDF counts AFTER computing this KF's vector
        if kf.bow_vec is not None:
            self._bow.add_document(kf.bow_vec)

        self.get_logger().debug(
            f"LC: keyframe {kf_id} at t={stamp:.3f}  "
            f"n_desc={len(descs)}  "
            f"bow_ready={self._bow.ready}  "
            f"n_3d={int(np.sum(~np.isnan(pts3d[:, 0]))) if pts3d is not None else 0}"
        )

        # Query for loop closures only when BoW is ready and we have enough KFs
        if self._bow.ready and kf.bow_vec is not None and kf_id >= self._min_kf_sep:
            self._detect_and_close_loop(kf, img0)

    @staticmethod
    def _decode_image(msg: Image) -> np.ndarray:
        """Convert a sensor_msgs/Image to a grayscale uint8 numpy array."""
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ("mono8", "8UC1"):
            return raw.reshape(msg.height, msg.width)
        img = raw.reshape(msg.height, msg.width, -1)
        code = cv2.COLOR_RGB2GRAY if msg.encoding == "rgb8" else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)

    def _compute_pts3d(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        kps: List[cv2.KeyPoint],
    ) -> Optional[np.ndarray]:
        """
        Compute a (N,3) array of 3-D positions in the rectified-cam0 frame for
        each ORB keypoint.  Entries are NaN where stereo depth is unavailable.

        Requires self._R_BS to be set (i.e. config_path was provided).
        """
        if self._R_BS is None or self._bf == 0.0:
            return None  # no stereo config — depth unavailable

        try:
            disp_raw = self._stereo_matcher.compute(img0, img1)
            disp = disp_raw.astype(np.float32) / 16.0  # SGBM returns fixed-point ×16
        except Exception:
            return None

        pts3d = np.full((len(kps), 3), np.nan, dtype=np.float64)
        h, w = disp.shape
        for i, kp in enumerate(kps):
            u = int(round(kp.pt[0]))
            v = int(round(kp.pt[1]))
            if not (0 <= u < w and 0 <= v < h):
                continue
            d = float(disp[v, u])
            if d < 1.0:  # sub-pixel or invalid disparity
                continue
            z = self._bf / d
            if z > 30.0:  # depth sanity cap at 30 m
                continue
            x = (u - self._cx_rect) * z / self._fx_rect
            y = (v - self._cy_rect) * z / self._fx_rect
            pts3d[i] = [x, y, z]
        return pts3d

    # ── Keyframe selection ─────────────────────────────────────────────────────

    def _should_be_keyframe(self, stamp: float) -> bool:
        """Decide whether the current frame should be a keyframe."""
        if self._last_kf_p is None:
            return True  # always accept the first frame

        # Time-based trigger
        dt = stamp - self._last_kf_stamp
        if dt >= self._kf_max_age:
            return True

        # Translation trigger
        dp = np.linalg.norm(self._latest_p - self._last_kf_p)
        if dp >= self._kf_trans_thr:
            return True

        # Rotation trigger
        dR = self._latest_R @ self._last_kf_R.T
        dtheta = np.linalg.norm(log_so3(dR))
        if dtheta >= self._kf_rot_thr_rad:
            return True

        return False

    # ── Vocabulary building ────────────────────────────────────────────────────

    def _build_vocabulary(self) -> None:
        """Build the k-means visual vocabulary from accumulated descriptors."""
        t0 = time.monotonic()
        all_descs = np.vstack(self._desc_pool)
        self.get_logger().info(
            f"LC: building BoW vocabulary ({self._bow._vocab_size} words) "
            f"from {len(all_descs)} descriptors across {len(self._desc_pool)} keyframes ..."
        )
        try:
            self._bow.build(all_descs)
        except Exception as exc:
            self.get_logger().warn(f"LC: vocabulary build failed: {exc}")
            return

        if self._bow.ready:
            elapsed = time.monotonic() - t0
            self.get_logger().info(
                f"LC: vocabulary built in {elapsed:.2f} s — "
                f"BoW retrieval now active"
            )
            # Retroactively compute BoW vectors for all existing keyframes
            for kf in self._keyframes:
                kf.bow_vec = self._bow.compute_bow(kf.descriptors)
                self._bow.add_document(kf.bow_vec)
            self._vocab_built = True
        else:
            self.get_logger().warn("LC: vocabulary build returned not-ready")

    # ── Loop detection ─────────────────────────────────────────────────────────

    def _detect_and_close_loop(
        self, query_kf: Keyframe, query_img: np.ndarray
    ) -> None:
        """
        Query the BoW database for loop candidates and attempt geometric
        verification.  If a loop is confirmed, run pose graph optimization.
        """
        best_score = -1.0
        best_idx = -1

        # Only search keyframes older than min_kf_sep
        search_end = len(self._keyframes) - self._min_kf_sep
        if search_end <= 0:
            return

        for idx in range(search_end):
            kf = self._keyframes[idx]
            if kf.bow_vec is None:
                continue
            score = self._bow.score(query_kf.bow_vec, kf.bow_vec)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score < self._min_score or best_idx < 0:
            return

        self.get_logger().info(
            f"LC: BoW candidate: query KF {query_kf.kf_id} → "
            f"DB KF {best_idx}  score={best_score:.4f}"
        )

        # Geometric verification
        db_kf = self._keyframes[best_idx]
        verified, T_rel = self._geometric_verify(query_kf, db_kf)
        if not verified:
            self.get_logger().debug(
                f"LC: geometric verification FAILED for KF pair "
                f"({query_kf.kf_id}, {best_idx})"
            )
            return

        # Sanity check: loop correction should not require a huge position jump
        t_corr = np.linalg.norm(T_rel[:3, 3])
        if t_corr > self._max_corr_m:
            self.get_logger().warn(
                f"LC: loop correction {t_corr:.2f} m exceeds "
                f"max_lc_correction_m={self._max_corr_m} m — rejected"
            )
            return

        self.get_logger().info(
            f"LC: loop CONFIRMED: KF {query_kf.kf_id} ↔ KF {best_idx}  "
            f"score={best_score:.4f}  |t|={t_corr:.3f} m"
        )

        # Add loop closure edge
        lc_edge = PoseGraphEdge(
            i=best_idx,
            j=query_kf.kf_id,
            T_ij=T_rel,
            sqrt_info=self._lc_sqrt_info.copy(),
            is_loop=True,
        )
        self._pg_edges.append(lc_edge)
        self._lc_count += 1

        # Run pose graph optimization
        self._run_pgo_and_publish(query_kf)

    # ── Geometric verification ─────────────────────────────────────────────────

    def _geometric_verify(
        self,
        query_kf: Keyframe,
        db_kf: Keyframe,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Verify a loop candidate and recover a metric body-frame relative pose.

        Strategy
        --------
        If stereo depth is available for query_kf (pts3d not None), use 3D-2D
        PnP RANSAC: match ORB keypoints, collect (3D_query, 2D_db) pairs for
        matches with valid depth, run solvePnPRansac, then convert the result
        from camera frame to body frame using the cam0↔body extrinsic R_BS.

        Falls back to Essential-matrix verification (no metric scale) when
        stereo depth is unavailable (no config_path was given).

        Returns (verified: bool, T_rel: 4×4 or None).
        T_rel encodes T_ij = T_db^{-1} @ T_query in body frame for the PGO.
        """
        # ── Descriptor matching ─────────────────────────────────────────────
        try:
            raw_matches = self._bf_matcher.knnMatch(
                query_kf.descriptors, db_kf.descriptors, k=2
            )
        except Exception as exc:
            self.get_logger().debug(f"LC: matching error: {exc}")
            return False, None

        good_matches = []
        for m_list in raw_matches:
            if len(m_list) < 2:
                continue
            m, n = m_list[0], m_list[1]
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < self._min_inliers:
            self.get_logger().info(
                f"LC: geom verify FAIL — too few ratio-test matches: "
                f"{len(good_matches)} < {self._min_inliers}"
            )
            return False, None

        # ── Stereo PnP path (preferred) ─────────────────────────────────────
        if (
            query_kf.pts3d is not None
            and self._R_BS is not None
            and self._K_rect is not None
        ):
            # Collect (3D query, 2D db) correspondences with valid depth
            pts3d_list: List[np.ndarray] = []
            pts2d_list: List[Tuple[float, float]] = []
            for m in good_matches:
                p3d = query_kf.pts3d[m.queryIdx]
                if not np.any(np.isnan(p3d)):
                    pts3d_list.append(p3d)
                    pts2d_list.append(db_kf.keypoints[m.trainIdx].pt)

            if len(pts3d_list) >= self._min_inliers:
                pts3d = np.array(pts3d_list, dtype=np.float64)
                pts2d = np.array(pts2d_list, dtype=np.float64)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts3d, pts2d, self._K_rect, None,
                    iterationsCount=200,
                    reprojectionError=4.0,
                    confidence=0.99,
                )
                if success and inliers is not None and len(inliers) >= self._min_inliers:
                    R_pnp, _ = cv2.Rodrigues(rvec)
                    t_pnp = tvec.ravel()

                    # Convert camera-frame result to body-frame T_ij.
                    # Derivation (R_BS maps cam0 vectors → body vectors):
                    #   R_ij_body = R_BS @ R_pnp @ R_BS.T
                    #   t_ij_body = R_BS @ t_pnp
                    R_ij = self._R_BS @ R_pnp @ self._R_BS.T
                    t_ij = self._R_BS @ t_pnp

                    T_rel = np.eye(4, dtype=np.float64)
                    T_rel[:3, :3] = R_ij
                    T_rel[:3, 3]  = t_ij

                    self.get_logger().debug(
                        f"LC: PnP verify OK: {len(inliers)} inliers  "
                        f"|t|={np.linalg.norm(t_ij):.3f} m"
                    )
                    return True, T_rel

                self.get_logger().info(
                    f"LC: geom verify FAIL — PnP inliers "
                    f"{'N/A' if not success else len(inliers) if inliers is not None else 0}"
                    f" < {self._min_inliers}"
                    f" (of {len(pts3d_list)} 3D-2D pairs)"
                )
                return False, None

            # Not enough 3D points — fall through to E-matrix
            self.get_logger().debug(
                f"LC: only {len(pts3d_list)} 3D points in query KF, "
                "falling back to E-matrix"
            )

        # ── E-matrix fallback (no metric scale) ─────────────────────────────
        if self._K_rect is None:
            self.get_logger().warn("LC: K_rect not yet available — skipping verification")
            return False, None

        query_pts = np.array(
            [query_kf.keypoints[m.queryIdx].pt for m in good_matches], dtype=np.float32
        )
        db_pts = np.array(
            [db_kf.keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float32
        )

        try:
            E, mask = cv2.findEssentialMat(
                query_pts, db_pts,
                cameraMatrix=self._K_rect,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
        except Exception as exc:
            self.get_logger().debug(f"LC: Essential matrix failed: {exc}")
            return False, None

        if E is None or mask is None:
            return False, None

        n_inliers = int(mask.sum())
        if n_inliers < self._min_inliers:
            self.get_logger().info(
                f"LC: geom verify FAIL — E-mat inliers={n_inliers} < min={self._min_inliers} "
                f"(of {len(good_matches)} ratio-test matches)"
            )
            return False, None

        # E-matrix verified — use ESKF-derived relative pose for the PGO edge
        # (no metric scale available without stereo depth).
        T_rel = LoopClosureNode._relative_pose(db_kf.p, db_kf.R, query_kf.p, query_kf.R)

        self.get_logger().debug(
            f"LC: E-mat verify OK: {n_inliers} inliers (no metric scale)"
        )
        return True, T_rel

    # ── Pose graph optimization ────────────────────────────────────────────────

    def _run_pgo_and_publish(self, latest_kf: Keyframe) -> None:
        """
        Run pose graph optimization and publish the corrected state.

        After PGO, compute the rigid correction:
            dT = T_new_latest @ T_old_latest^{-1}
        and publish it on /lc/correction so the ESKF can apply it.
        """
        n = len(self._keyframes)
        if n == 0:
            return

        # Initial poses for optimizer
        init_poses = [(kf.p.copy(), kf.R.copy()) for kf in self._keyframes]

        t0 = time.monotonic()
        try:
            opt_poses = self._pgo.optimize(init_poses, self._pg_edges)
        except Exception as exc:
            self.get_logger().warn(
                f"LC: PGO failed: {exc} — skipping correction"
            )
            return
        elapsed = time.monotonic() - t0

        self.get_logger().info(
            f"LC: PGO converged in {elapsed*1000:.1f} ms  "
            f"({n} nodes, {len(self._pg_edges)} edges, "
            f"loop #{self._lc_count})"
        )

        # Compute correction transform for the ESKF.
        # Do NOT update stored keyframe poses — they must stay in ESKF coordinates
        # so that odometry edges built from consecutive keyframes remain consistent.
        # Permanently shifting keyframe poses while ESKF hasn't been corrected yet
        # would encode the rejected correction into newly created odometry edges,
        # causing runaway divergence on subsequent PGO runs.
        idx_latest = latest_kf.kf_id
        p_new, R_new = opt_poses[idx_latest]
        p_old_init, R_old_init = init_poses[idx_latest]
        dp = p_new - p_old_init
        dR = R_new @ R_old_init.T

        correction_too_large = np.linalg.norm(dp) > self._max_corr_m
        if correction_too_large:
            self.get_logger().warn(
                f"LC: PGO correction {np.linalg.norm(dp):.2f} m exceeds limit — "
                f"applying but flagging"
            )

        # Publish correction
        self._publish_correction(dp, dR, latest_kf.stamp)

        # Publish global corrected trajectory
        self._publish_global_trajectory()

    # ── Publishers ─────────────────────────────────────────────────────────────

    def _publish_correction(
        self,
        dp: np.ndarray,
        dR: np.ndarray,
        stamp_sec: float,
    ) -> None:
        """
        Publish the SE(3) correction on /lc/correction.

        The ESKF handler will apply:
            p_new = p_eskf + dp
            R_new = dR @ R_eskf
        """
        msg = PoseStamped()

        # Encode stamp
        stamp_sec_int = int(stamp_sec)
        stamp_nsec = int((stamp_sec - stamp_sec_int) * 1e9)
        msg.header.stamp.sec = stamp_sec_int
        msg.header.stamp.nanosec = stamp_nsec
        msg.header.frame_id = "map"

        # Position field carries the position CORRECTION delta_p
        msg.pose.position.x = float(dp[0])
        msg.pose.position.y = float(dp[1])
        msg.pose.position.z = float(dp[2])

        # Orientation field carries the rotation correction as a quaternion
        q_corr = rot_to_quat(dR)
        msg.pose.orientation.x = float(q_corr[0])
        msg.pose.orientation.y = float(q_corr[1])
        msg.pose.orientation.z = float(q_corr[2])
        msg.pose.orientation.w = float(q_corr[3])

        self._pub_correction.publish(msg)
        self.get_logger().info(
            f"LC: correction published: dp={np.round(dp, 3)}  "
            f"|dp|={np.linalg.norm(dp):.3f} m  "
            f"|dtheta|={np.degrees(np.linalg.norm(log_so3(dR))):.2f} deg"
        )

    def _publish_global_trajectory(self) -> None:
        """Publish all optimized keyframe poses as a nav_msgs/Path."""
        path = Path()
        path.header.frame_id = "map"

        for kf in self._keyframes:
            ps = PoseStamped()
            stamp_sec_int = int(kf.stamp)
            ps.header.stamp.sec = stamp_sec_int
            ps.header.stamp.nanosec = int((kf.stamp - stamp_sec_int) * 1e9)
            ps.header.frame_id = "map"
            ps.pose.position.x = float(kf.p[0])
            ps.pose.position.y = float(kf.p[1])
            ps.pose.position.z = float(kf.p[2])
            q = rot_to_quat(kf.R)
            ps.pose.orientation.x = float(q[0])
            ps.pose.orientation.y = float(q[1])
            ps.pose.orientation.z = float(q[2])
            ps.pose.orientation.w = float(q[3])
            path.poses.append(ps)

        if path.poses:
            path.header.stamp = path.poses[-1].header.stamp

        self._pub_path.publish(path)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _relative_pose(
        p_i: np.ndarray, R_i: np.ndarray,
        p_j: np.ndarray, R_j: np.ndarray,
    ) -> np.ndarray:
        """Compute 4×4 relative transform T_ij = T_i^{-1} @ T_j."""
        T_ij = np.eye(4, dtype=np.float64)
        T_ij[:3, :3] = R_i.T @ R_j
        T_ij[:3, 3] = R_i.T @ (p_j - p_i)
        return T_ij


# ── Entry point ────────────────────────────────────────────────────────────────


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LoopClosureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
