#!/usr/bin/env python3
"""
factor_graph.py
===============
Sliding-Window Factor Graph backend for Visual-Inertial Odometry.

Pure NumPy — no ROS dependency.

This is a Stage A implementation: the factor graph contains only keyframe pose/
velocity/IMU states.  There are no landmark variables (point-cloud factors are
deferred to Stage B).

State layout per keyframe (KF_DIM = 15)
-----------------------------------------
  [dp(3) | dv(3) | dtheta(3) | db_a(3) | db_g(3)]
   ^position ^velocity ^SO(3) tangent ^accel bias ^gyro bias

Factors implemented
--------------------
  1. IMU preintegration factors   (9-DOF residual between consecutive KFs)
  2. VO absolute-pose anchors     (6-DOF residual from visual odometry)
  3. Velocity priors              (3-DOF soft constraint on velocity)
  4. Bias random-walk factors     (6-DOF between consecutive KF biases)
  5. Marginalization prior        (Schur complement of oldest KF)

Optimizer: Levenberg-Marquardt (Gauss-Newton with adaptive damping).

References
----------
Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial
Odometry," TRO 2017.
"""

import numpy as np
from vio_pipeline.imu_preintegrator import (
    _skew,
    exp_so3,
    log_so3,
    quat_to_rot,
    rot_to_quat,
    quat_mul,
    ImuPreintegrator,
)

# Dimension of one keyframe's tangent-space state vector
KF_DIM: int = 15  # [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]

# Minimum covariance floor for IMU factors to prevent ill-conditioning.
# Values correspond to 1-sigma: pos≈0.1 m, vel≈0.3 m/s, angle≈0.5°
# Position floor raised to 1e-2 so IMU info (~100) is comparable to VO info
# (sigma_trans=0.05 → info=400) rather than dominating by 25×.
_COV_FLOOR = np.diag(
    np.array(
        [1e-2, 1e-2, 1e-2,    # delta_p  [m^2]  (σ≈0.1 m)
         0.09, 0.09, 0.09,    # delta_v  [m^2/s^2]
         7.6e-5, 7.6e-5, 7.6e-5],  # delta_theta [rad^2]  (~0.5 deg)
        dtype=np.float64,
    )
)


# ──────────────────────────────────────────────────────────────────────────────
#  SO(3) ↔ quaternion helper
# ──────────────────────────────────────────────────────────────────────────────


def _exp_so3_quat(phi: np.ndarray) -> np.ndarray:
    """Rotation vector phi → quaternion [x, y, z, w].

    angle = ||phi||
    For angle < 1e-10 (small angle): q ≈ [phi/2, 1] normalized
    Else:  q = [sin(angle/2) * axis, cos(angle/2)]
    """
    phi = np.asarray(phi, dtype=np.float64)
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        half = phi * 0.5
        q = np.array([half[0], half[1], half[2], 1.0], dtype=np.float64)
        return q / np.linalg.norm(q)
    axis = phi / angle
    s = np.sin(angle * 0.5)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle * 0.5)],
        dtype=np.float64,
    )


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion; return identity if norm is near zero."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


# ──────────────────────────────────────────────────────────────────────────────
#  KeyframeState
# ──────────────────────────────────────────────────────────────────────────────


class KeyframeState:
    """6-DOF pose + velocity + IMU biases for a single keyframe.

    Attributes
    ----------
    p       : (3,)  Position in world frame      [m]
    v       : (3,)  Velocity in world frame       [m/s]
    q       : (4,)  Orientation [x,y,z,w]
    b_a     : (3,)  Accelerometer bias            [m/s^2]
    b_g     : (3,)  Gyroscope bias                [rad/s]
    stamp_ns: int   Timestamp in nanoseconds
    """

    def __init__(
        self,
        p,
        v,
        q,
        b_a,
        b_g,
        stamp_ns: int = 0,
    ) -> None:
        self.p = np.asarray(p, dtype=np.float64).copy()
        self.v = np.asarray(v, dtype=np.float64).copy()
        self.q = _normalize_quat(np.asarray(q, dtype=np.float64))
        self.b_a = np.asarray(b_a, dtype=np.float64).copy()
        self.b_g = np.asarray(b_g, dtype=np.float64).copy()
        self.stamp_ns = int(stamp_ns)

    def copy(self) -> "KeyframeState":
        """Return a deep copy of this state."""
        return KeyframeState(
            p=self.p.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            b_a=self.b_a.copy(),
            b_g=self.b_g.copy(),
            stamp_ns=self.stamp_ns,
        )

    def retract(self, dx: np.ndarray) -> None:
        """Apply a tangent-space perturbation dx ∈ R^15 to this state in place.

        Layout: [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]

        Position and velocity are updated additively.
        Orientation uses a right-multiplication on SO(3):
            q ← normalize(q ⊗ exp_quat(dtheta))
        Biases are updated additively.
        """
        dx = np.asarray(dx, dtype=np.float64)
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.q = _normalize_quat(quat_mul(self.q, _exp_so3_quat(dx[6:9])))
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]


# ──────────────────────────────────────────────────────────────────────────────
#  SlidingWindowGraph
# ──────────────────────────────────────────────────────────────────────────────


class SlidingWindowGraph:
    """Sliding-window factor graph with Levenberg-Marquardt optimization.

    The graph maintains a fixed-size window of keyframes.  When the window
    overflows, the oldest keyframe is marginalized via Schur complement and
    its information is preserved as a dense prior on the remaining states.

    Parameters
    ----------
    gravity     : (3,)  Gravity vector in world frame [m/s^2], e.g. [0,0,-9.81]
    window_size : int   Maximum number of keyframes before marginalization
    """

    def __init__(self, gravity, window_size: int = 10) -> None:
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.window_size = int(window_size)

        # Keyframe storage — parallel lists, same index
        self._keyframes: list[KeyframeState] = []
        self._kf_ids: list[int] = []
        self._next_kf_id: int = 0

        # IMU factors: kf_id_i → (kf_id_j, ImuPreintegrator)
        # The factor connects consecutive keyframes i and j=i+1.
        self._imu_factors: dict[int, tuple[int, ImuPreintegrator]] = {}

        # VO absolute-pose anchors: (kf_id, p_meas, q_meas, sigma_t, sigma_r)
        self._pose_factors: list[tuple] = []

        # Velocity priors: (kf_id, v_meas, sigma_v)
        self._vel_priors: list[tuple] = []

        # Marginalization prior (Schur complement from oldest KF)
        self._prior_H: np.ndarray | None = None   # (n_remaining*KF_DIM, n_remaining*KF_DIM)
        self._prior_b: np.ndarray | None = None   # (n_remaining*KF_DIM,)
        self._prior_kf_ids: list[int] = []        # which kf_ids the prior references

        # Bias random-walk noise (used in _assemble_hessian)
        self._sigma_ba: float = 3.0e-3
        self._sigma_bg: float = 1.9393e-5

    # ── Public interface ───────────────────────────────────────────────────────

    def add_keyframe(
        self,
        state: KeyframeState,
        preintegrator: ImuPreintegrator | None = None,
    ) -> int:
        """Add a keyframe to the window.

        If a preintegrator is provided and there is at least one existing
        keyframe, an IMU factor is stored between the previous keyframe and
        the new one.

        If the window exceeds window_size after insertion, the oldest keyframe
        is marginalized.

        Parameters
        ----------
        state         : KeyframeState  Initial state estimate for the new KF
        preintegrator : ImuPreintegrator or None

        Returns
        -------
        kf_id : int  Unique ID assigned to the new keyframe
        """
        kf_id = self._next_kf_id
        self._next_kf_id += 1

        self._keyframes.append(state.copy())
        self._kf_ids.append(kf_id)

        # Store IMU factor between previous KF and this one
        if preintegrator is not None and len(self._keyframes) >= 2:
            prev_id = self._kf_ids[-2]
            self._imu_factors[prev_id] = (kf_id, preintegrator)

        # Marginalize if over capacity
        if len(self._keyframes) > self.window_size:
            self._marginalize_oldest()

        return kf_id

    def add_pose_factor(
        self,
        kf_id: int,
        p_meas,
        q_meas,
        sigma_trans: float = 0.05,
        sigma_rot: float = 0.05,
    ) -> None:
        """Add a VO absolute-pose anchor for the given keyframe.

        Residual: r = [p_kf - p_meas;  log_so3(R_meas.T @ R_kf)]  ∈ R^6

        Parameters
        ----------
        kf_id      : int    Target keyframe ID
        p_meas     : (3,)   Measured position        [m]
        q_meas     : (4,)   Measured orientation     [x,y,z,w]
        sigma_trans: float  Translation noise std    [m]
        sigma_rot  : float  Rotation noise std       [rad]
        """
        self._pose_factors.append((
            kf_id,
            np.asarray(p_meas, dtype=np.float64).copy(),
            _normalize_quat(np.asarray(q_meas, dtype=np.float64)),
            float(sigma_trans),
            float(sigma_rot),
        ))

    def add_velocity_prior(
        self,
        kf_id: int,
        v_meas,
        sigma_v: float = 0.5,
    ) -> None:
        """Add a soft velocity prior for the given keyframe.

        Residual: r = v_kf - v_meas  ∈ R^3

        Parameters
        ----------
        kf_id   : int   Target keyframe ID
        v_meas  : (3,)  Measured/expected velocity [m/s]
        sigma_v : float Velocity noise std         [m/s]
        """
        self._vel_priors.append((
            kf_id,
            np.asarray(v_meas, dtype=np.float64).copy(),
            float(sigma_v),
        ))

    def get_keyframe(self, kf_id: int) -> KeyframeState | None:
        """Return a copy of the KeyframeState with the given ID, or None."""
        for i, kid in enumerate(self._kf_ids):
            if kid == kf_id:
                return self._keyframes[i].copy()
        return None

    def latest_keyframe(self) -> tuple[int | None, KeyframeState | None]:
        """Return (kf_id, KeyframeState) of the most recent keyframe.

        Returns (None, None) if the window is empty.
        """
        if not self._keyframes:
            return None, None
        return self._kf_ids[-1], self._keyframes[-1].copy()

    def num_keyframes(self) -> int:
        """Return the current number of keyframes in the window."""
        return len(self._keyframes)

    # ── Residual computation ───────────────────────────────────────────────────

    def _imu_residual(
        self,
        kf_i: KeyframeState,
        kf_j: KeyframeState,
        preint: ImuPreintegrator,
    ) -> np.ndarray:
        """Compute the 9-DOF IMU preintegration residual in the body frame at i.

        From Forster TRO 2017:
            r_p = R_i^T (p_j - p_i - v_i*dt - 0.5*g*dt^2) - delta_p_corr
            r_v = R_i^T (v_j - v_i - g*dt) - delta_v_corr
            r_R = log_so3(delta_R_corr^T @ R_i^T @ R_j)

        Parameters
        ----------
        kf_i, kf_j : KeyframeState  Keyframes at start and end of interval
        preint      : ImuPreintegrator

        Returns
        -------
        residual : (9,) [r_p(3), r_v(3), r_R(3)]
        """
        dt = preint.dt_total
        R_i = quat_to_rot(kf_i.q)
        R_j = quat_to_rot(kf_j.q)
        g = self.gravity

        dp_corr, dv_corr, dR_corr = preint.bias_corrected_measurement(kf_i.b_a, kf_i.b_g)

        r_p = R_i.T @ (kf_j.p - kf_i.p - kf_i.v * dt - 0.5 * g * dt * dt) - dp_corr
        r_v = R_i.T @ (kf_j.v - kf_i.v - g * dt) - dv_corr
        r_R = log_so3(dR_corr.T @ R_i.T @ R_j)

        return np.concatenate([r_p, r_v, r_R])

    def _imu_jacobians_numerical(
        self,
        kf_i: KeyframeState,
        kf_j: KeyframeState,
        preint: ImuPreintegrator,
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute IMU factor Jacobians via forward finite differences.

        Returns
        -------
        J_i : (9, 15)  Jacobian w.r.t. state i
        J_j : (9, 15)  Jacobian w.r.t. state j
        """
        r0 = self._imu_residual(kf_i, kf_j, preint)
        n_res = len(r0)  # 9

        J_i = np.zeros((n_res, KF_DIM), dtype=np.float64)
        J_j = np.zeros((n_res, KF_DIM), dtype=np.float64)

        # Perturb kf_i
        for k in range(KF_DIM):
            dx = np.zeros(KF_DIM, dtype=np.float64)
            dx[k] = eps
            kf_i_p = kf_i.copy()
            kf_i_p.retract(dx)
            r_p = self._imu_residual(kf_i_p, kf_j, preint)
            J_i[:, k] = (r_p - r0) / eps

        # Perturb kf_j
        for k in range(KF_DIM):
            dx = np.zeros(KF_DIM, dtype=np.float64)
            dx[k] = eps
            kf_j_p = kf_j.copy()
            kf_j_p.retract(dx)
            r_p = self._imu_residual(kf_i, kf_j_p, preint)
            J_j[:, k] = (r_p - r0) / eps

        return J_i, J_j

    # ── Hessian assembly ───────────────────────────────────────────────────────

    def _kf_index(self, kf_id: int) -> int | None:
        """Return the 0-based position of kf_id in self._kf_ids, or None."""
        for i, kid in enumerate(self._kf_ids):
            if kid == kf_id:
                return i
        return None

    def _kf_slice(self, kf_idx: int) -> slice:
        """Return the slice into the full state vector for keyframe index kf_idx."""
        start = kf_idx * KF_DIM
        return slice(start, start + KF_DIM)

    def _assemble_hessian(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Build the Gauss-Newton Hessian H and gradient b over all factors.

        State ordering: [kf_0(15) | kf_1(15) | ... | kf_{N-1}(15)]

        Returns
        -------
        H    : (total_dim, total_dim)  Approximate Hessian (J^T W J)
        b    : (total_dim,)            Gradient (J^T W r)
        cost : float                   0.5 * sum(r^T Sigma_inv r)
        """
        n_kf = len(self._keyframes)
        if n_kf == 0:
            return np.zeros((0, 0)), np.zeros(0), 0.0

        total_dim = n_kf * KF_DIM
        H = np.zeros((total_dim, total_dim), dtype=np.float64)
        b = np.zeros(total_dim, dtype=np.float64)
        cost = 0.0

        # ── 1. Marginalization prior ───────────────────────────────────────
        self._add_prior_to_hessian(H, b)

        # ── 2. VO absolute-pose factors ────────────────────────────────────
        for kf_id, p_meas, q_meas, sigma_t, sigma_r in self._pose_factors:
            idx = self._kf_index(kf_id)
            if idx is None:
                continue
            kf = self._keyframes[idx]
            R_kf = quat_to_rot(kf.q)
            R_meas = quat_to_rot(q_meas)

            # Residual
            r_p = kf.p - p_meas                      # (3,)
            r_r = log_so3(R_meas.T @ R_kf)            # (3,)
            r = np.concatenate([r_p, r_r])             # (6,)

            # Information (diagonal, isotropic blocks)
            info_t = 1.0 / (sigma_t * sigma_t)
            info_r = 1.0 / (sigma_r * sigma_r)
            Sigma_inv = np.diag(
                np.array([info_t] * 3 + [info_r] * 3, dtype=np.float64)
            )

            # Jacobian J (6, 15): identity blocks at position and angle slots
            J = np.zeros((6, KF_DIM), dtype=np.float64)
            J[0:3, 0:3] = np.eye(3)   # d(r_p)/d(delta_p)
            J[3:6, 6:9] = np.eye(3)   # d(r_r)/d(delta_theta)

            # Accumulate Hessian block for this keyframe
            sl = self._kf_slice(idx)
            JtW = J.T @ Sigma_inv
            H[sl, sl] += JtW @ J
            b[sl] += JtW @ r
            cost += 0.5 * r @ Sigma_inv @ r

        # ── 3. Velocity priors ─────────────────────────────────────────────
        for kf_id, v_meas, sigma_v in self._vel_priors:
            idx = self._kf_index(kf_id)
            if idx is None:
                continue
            kf = self._keyframes[idx]
            r = kf.v - v_meas                          # (3,)
            info_v = 1.0 / (sigma_v * sigma_v)
            Sigma_inv = info_v * np.eye(3, dtype=np.float64)

            # Jacobian: d(r)/d(delta_v) = I at v block [3:6]
            J = np.zeros((3, KF_DIM), dtype=np.float64)
            J[0:3, 3:6] = np.eye(3)

            sl = self._kf_slice(idx)
            JtW = J.T @ Sigma_inv
            H[sl, sl] += JtW @ J
            b[sl] += JtW @ r
            cost += 0.5 * r @ Sigma_inv @ r

        # ── 4. IMU preintegration factors ─────────────────────────────────
        for kf_id_i, (kf_id_j, preint) in self._imu_factors.items():
            idx_i = self._kf_index(kf_id_i)
            idx_j = self._kf_index(kf_id_j)
            if idx_i is None or idx_j is None:
                continue

            kf_i = self._keyframes[idx_i]
            kf_j = self._keyframes[idx_j]

            r = self._imu_residual(kf_i, kf_j, preint)  # (9,)

            # Apply covariance floor before inversion
            Cov_safe = np.maximum(preint.Cov, _COV_FLOOR) + 1e-8 * np.eye(9)
            Cov_safe = 0.5 * (Cov_safe + Cov_safe.T)  # enforce symmetry
            Sigma_inv = np.linalg.inv(Cov_safe)
            Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)

            # Numerical Jacobians (9×15 each)
            J_i, J_j = self._imu_jacobians_numerical(kf_i, kf_j, preint)

            sl_i = self._kf_slice(idx_i)
            sl_j = self._kf_slice(idx_j)

            JiT_W = J_i.T @ Sigma_inv
            JjT_W = J_j.T @ Sigma_inv

            H[sl_i, sl_i] += JiT_W @ J_i
            H[sl_i, sl_j] += JiT_W @ J_j
            H[sl_j, sl_i] += JjT_W @ J_i
            H[sl_j, sl_j] += JjT_W @ J_j

            b[sl_i] += JiT_W @ r
            b[sl_j] += JjT_W @ r
            cost += 0.5 * r @ Sigma_inv @ r

            # ── 4b. Bias random-walk between consecutive KFs ───────────────
            dt = preint.dt_total
            if dt > 1e-9:
                # Accel bias walk
                info_ba = 1.0 / (self._sigma_ba * self._sigma_ba * dt)
                r_ba = kf_j.b_a - kf_i.b_a              # (3,)
                # J_ba_i: -I at db_a block of kf_i; J_ba_j: +I at db_a block of kf_j
                H[sl_i, sl_i][9:12, 9:12] += info_ba * np.eye(3)
                H[sl_j, sl_j][9:12, 9:12] += info_ba * np.eye(3)
                H[sl_i, sl_j][9:12, 9:12] -= info_ba * np.eye(3)
                H[sl_j, sl_i][9:12, 9:12] -= info_ba * np.eye(3)
                b[sl_i][9:12] -= info_ba * r_ba
                b[sl_j][9:12] += info_ba * r_ba
                cost += 0.5 * info_ba * np.dot(r_ba, r_ba)

                # Gyro bias walk
                info_bg = 1.0 / (self._sigma_bg * self._sigma_bg * dt)
                r_bg = kf_j.b_g - kf_i.b_g              # (3,)
                H[sl_i, sl_i][12:15, 12:15] += info_bg * np.eye(3)
                H[sl_j, sl_j][12:15, 12:15] += info_bg * np.eye(3)
                H[sl_i, sl_j][12:15, 12:15] -= info_bg * np.eye(3)
                H[sl_j, sl_i][12:15, 12:15] -= info_bg * np.eye(3)
                b[sl_i][12:15] -= info_bg * r_bg
                b[sl_j][12:15] += info_bg * r_bg
                cost += 0.5 * info_bg * np.dot(r_bg, r_bg)

        return H, b, cost

    def _add_prior_to_hessian(self, H: np.ndarray, b: np.ndarray) -> None:
        """Inject the stored marginalization prior into H and b.

        The prior references a subset of the current window.  If any of the
        prior's target KF IDs are no longer in the window, the prior is skipped
        (this can happen during graph reset or after multiple marginalizations
        in quick succession).

        Parameters
        ----------
        H : (total_dim, total_dim)  Hessian to update in place
        b : (total_dim,)            Gradient to update in place
        """
        if self._prior_H is None or self._prior_b is None:
            return

        # Build index mapping from prior KF IDs to positions in the current window.
        # Skip any KF IDs that have since been marginalized out — only scatter
        # the blocks that are still present (partial prior is better than none).
        prior_kf_indices = []   # index in current window
        valid_prior_pos = []    # corresponding block index in prior arrays
        for pi, pid in enumerate(self._prior_kf_ids):
            idx = self._kf_index(pid)
            if idx is not None:
                prior_kf_indices.append(idx)
                valid_prior_pos.append(pi)

        if not prior_kf_indices:
            return  # None of the prior KFs are in the window any more

        # Scatter prior blocks into H and b
        for pi, idx_i in zip(valid_prior_pos, prior_kf_indices):
            sl_i = self._kf_slice(idx_i)
            b[sl_i] += self._prior_b[pi * KF_DIM:(pi + 1) * KF_DIM]
            for pj, idx_j in zip(valid_prior_pos, prior_kf_indices):
                sl_j = self._kf_slice(idx_j)
                H[sl_i, sl_j] += self._prior_H[
                    pi * KF_DIM:(pi + 1) * KF_DIM,
                    pj * KF_DIM:(pj + 1) * KF_DIM,
                ]

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def optimize(self, max_iters: int = 5, lm_lambda_init: float = 1e-4) -> float:
        """Levenberg-Marquardt optimizer.

        Each iteration:
          1. Assemble Gauss-Newton Hessian and gradient.
          2. Damp diagonal: H_damp = H + lambda * diag(max(H_ii, 1e-6)) + 1e-8*I
          3. Solve the linear system for the step dx.
          4. Tentatively retract all keyframes, compute new cost.
          5. Accept if cost decreased (reduce lambda); reject otherwise (increase lambda).

        Parameters
        ----------
        max_iters       : int    Maximum LM iterations
        lm_lambda_init  : float  Initial damping factor

        Returns
        -------
        cost : float  Final cost after optimization
        """
        if len(self._keyframes) == 0:
            return 0.0

        lm_lambda = float(lm_lambda_init)

        H, b, cost = self._assemble_hessian()

        for _ in range(max_iters):
            # Adaptive diagonal damping
            diag_H = np.maximum(np.diag(H), 1e-6)
            H_damp = H + lm_lambda * np.diag(diag_H) + 1e-8 * np.eye(len(H))

            # Solve H_damp @ dx = -b
            try:
                dx = np.linalg.solve(H_damp, -b)
            except np.linalg.LinAlgError:
                lm_lambda = min(lm_lambda * 10.0, 1e8)
                continue

            if not np.all(np.isfinite(dx)):
                lm_lambda = min(lm_lambda * 10.0, 1e8)
                continue

            # Backup current state
            backup = [kf.copy() for kf in self._keyframes]

            # Retract all keyframes
            for i, kf in enumerate(self._keyframes):
                kf.retract(dx[i * KF_DIM:(i + 1) * KF_DIM])

            # Evaluate new cost — reuse H_new, b_new on accept to avoid a
            # redundant third assembly of the Hessian per iteration.
            H_new, b_new, new_cost = self._assemble_hessian()

            if new_cost < cost:
                # Accept: step was good, reduce damping
                lm_lambda = max(lm_lambda / 3.0, 1e-8)
                H, b, cost = H_new, b_new, new_cost
            else:
                # Reject: restore backup and increase damping
                self._keyframes = backup
                lm_lambda = min(lm_lambda * 10.0, 1e8)

        return cost

    # ── Marginalization ────────────────────────────────────────────────────────

    def _marginalize_oldest(self) -> None:
        """Schur complement marginalization of the oldest keyframe.

        Builds the full Hessian, then marginalizes out the first KF_DIM rows/
        columns (the oldest keyframe's variables).  The resulting dense prior
        is stored and replaces any previous prior.

        After marginalization:
          - The oldest keyframe and its pose/vel factors are removed.
          - Any IMU factor whose source is the oldest KF is removed.
          - The marginalization prior is updated to reference kf_ids[1:].
        """
        if len(self._keyframes) < 2:
            return

        H_full, b_full, _ = self._assemble_hessian()

        # alpha: indices of the oldest keyframe (to be eliminated)
        alpha = slice(0, KF_DIM)
        # beta: indices of all remaining keyframes
        beta = slice(KF_DIM, H_full.shape[0])

        H_aa = H_full[alpha, alpha] + 1e-8 * np.eye(KF_DIM)
        H_ab = H_full[alpha, beta]
        H_ba = H_full[beta, alpha]
        H_bb = H_full[beta, beta]
        b_a = b_full[alpha]
        b_b = b_full[beta]

        # Schur complement: H_prior = H_bb - H_ba @ inv(H_aa) @ H_ab
        H_aa_inv = np.linalg.inv(H_aa)
        H_prior = H_bb - H_ba @ H_aa_inv @ H_ab
        b_prior = b_b - H_ba @ H_aa_inv @ b_a

        # Enforce symmetry
        H_prior = 0.5 * (H_prior + H_prior.T)

        # Store prior referencing kf_ids[1:]
        self._prior_H = H_prior
        self._prior_b = b_prior
        self._prior_kf_ids = list(self._kf_ids[1:])

        # Identify the oldest KF ID
        oldest_id = self._kf_ids[0]

        # Remove oldest keyframe's pose/vel factors
        self._pose_factors = [
            f for f in self._pose_factors if f[0] != oldest_id
        ]
        self._vel_priors = [
            f for f in self._vel_priors if f[0] != oldest_id
        ]

        # Remove IMU factor originating from oldest KF
        self._imu_factors.pop(oldest_id, None)

        # Remove the oldest keyframe from both parallel lists
        self._keyframes.pop(0)
        self._kf_ids.pop(0)
