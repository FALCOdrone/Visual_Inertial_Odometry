"""
factor_graph.py
===============
Sliding-window factor graph for Visual-Inertial Odometry (Stage A: pose-level).

Implements Levenberg-Marquardt optimization over a sliding window of keyframes
connected by IMU preintegration factors and (optionally) visual reprojection
factors.  Marginalization of the oldest keyframe uses Schur complement to
produce a prior factor that retains the information.

This is a standalone Python module with NO ROS dependency.

State per keyframe: x = [p(3), v(3), q(4), b_a(3), b_g(3)]
Tangent-space perturbation: delta_x in R^15 = [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]

References
----------
Forster et al., "IMU Preintegration on Manifold", TRO 2017.
Leutenegger et al., "Keyframe-based Visual-Inertial Odometry using Nonlinear
    Optimization", IJRR 2015.
"""
import numpy as np
from copy import deepcopy

from vio_pipeline.imu_preintegrator import (
    _skew, exp_so3, log_so3, right_jacobian_so3,
    quat_to_rot, rot_to_quat, quat_mul, ImuPreintegrator
)


# ---- Tangent-space dimension per keyframe ----
KF_DIM = 15   # [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]
LM_DIM = 3    # landmark position in world frame


def _exp_so3_quat(phi):
    """
    Rotation vector -> quaternion [x, y, z, w].

    q = [sin(theta/2) * axis; cos(theta/2)]
    For small theta uses first-order Taylor expansion.
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        half = phi * 0.5
        return np.array([half[0], half[1], half[2], 1.0], dtype=np.float64) / np.sqrt(
            1.0 + 0.25 * angle * angle)
    axis = phi / angle
    s = np.sin(angle * 0.5)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle*0.5)],
                    dtype=np.float64)


# ---- Keyframe state ---------------------------------------------------------

class KeyframeState:
    """
    State of a single keyframe in the sliding window.

    Attributes
    ----------
    p : ndarray (3,)
        Position in world frame [m].
    v : ndarray (3,)
        Velocity in world frame [m/s].
    q : ndarray (4,)
        Orientation quaternion [x, y, z, w] (body -> world).
    b_a : ndarray (3,)
        Accelerometer bias [m/s^2].
    b_g : ndarray (3,)
        Gyroscope bias [rad/s].
    stamp_ns : int
        Timestamp in nanoseconds.
    """

    def __init__(self, p, v, q, b_a, b_g, stamp_ns=0):
        self.p = np.array(p, dtype=np.float64)
        self.v = np.array(v, dtype=np.float64)
        self.q = np.array(q, dtype=np.float64)
        self.q /= np.linalg.norm(self.q)
        self.b_a = np.array(b_a, dtype=np.float64)
        self.b_g = np.array(b_g, dtype=np.float64)
        self.stamp_ns = int(stamp_ns)

    def copy(self):
        """Return a deep copy of this state."""
        return KeyframeState(
            self.p.copy(), self.v.copy(), self.q.copy(),
            self.b_a.copy(), self.b_g.copy(), self.stamp_ns)

    def retract(self, dx):
        """
        Apply a tangent-space perturbation dx in R^15 to update the state.

        dx layout: [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]

        Position and velocity are updated additively.
        Orientation is updated multiplicatively: q <- q * Exp(dtheta).
        Biases are updated additively.
        """
        self.p += dx[0:3]
        self.v += dx[3:6]
        dq = _exp_so3_quat(dx[6:9])
        self.q = quat_mul(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        self.b_a += dx[9:12]
        self.b_g += dx[12:15]


# ---- Landmark state ----------------------------------------------------------

class LandmarkState:
    """
    3D landmark position in world frame.

    Attributes
    ----------
    p : ndarray (3,)
        Position [m] in world frame.
    lm_id : int
        Unique landmark identifier.
    """

    def __init__(self, p, lm_id):
        self.p = np.array(p, dtype=np.float64)
        self.lm_id = int(lm_id)

    def copy(self):
        return LandmarkState(self.p.copy(), self.lm_id)


# ---- Sliding Window Factor Graph --------------------------------------------

class SlidingWindowGraph:
    """
    Sliding-window factor graph optimizer for VIO.

    Manages a window of keyframes connected by IMU preintegration factors
    and (optionally) visual reprojection factors.  Uses Levenberg-Marquardt
    for optimization and Schur complement for marginalization.

    Parameters
    ----------
    cam_K : ndarray (3, 3)
        Camera intrinsic matrix (rectified cam0).
    T_b_c0 : ndarray (4, 4)
        Rigid transform body <- cam0  (p_body = T_b_c0 @ p_cam0).
    gravity : ndarray (3,)
        Gravity vector in world frame [m/s^2], e.g. [0, 0, -9.81].
    window_size : int
        Maximum number of keyframes in the sliding window.
    """

    def __init__(self, cam_K, T_b_c0, gravity, window_size=10):
        self.cam_K = np.array(cam_K, dtype=np.float64)
        self.fx = self.cam_K[0, 0]
        self.fy = self.cam_K[1, 1]
        self.cx = self.cam_K[0, 2]
        self.cy = self.cam_K[1, 2]

        self.T_b_c0 = np.array(T_b_c0, dtype=np.float64)
        self.R_bc = self.T_b_c0[:3, :3]  # rotation body <- cam0
        self.t_bc = self.T_b_c0[:3, 3]   # translation body <- cam0

        self.gravity = np.array(gravity, dtype=np.float64)
        self.window_size = window_size

        # Ordered list of keyframe states and their unique IDs
        self._keyframes = []      # list of KeyframeState
        self._kf_ids = []         # parallel list of kf_id (int)
        self._next_kf_id = 0

        # IMU preintegration factors: dict kf_id_i -> (kf_id_j, preintegrator)
        self._imu_factors = {}

        # VO pose measurement factors: list of (kf_id, p_meas, q_meas, sigma_t, sigma_r)
        # These anchor keyframes to their VO pose estimates (Stage A)
        self._pose_factors = []

        # Velocity priors: list of (kf_id, v_meas, sigma_v)
        # Anchors velocity to prevent unbounded null-space drift
        self._vel_priors = []

        # Visual factors: list of (kf_id, lm_id, uv_obs)
        self._vis_factors = []

        # Landmarks: dict lm_id -> LandmarkState
        self._landmarks = {}

        # Marginalization prior: accumulated from previous window shifts
        self._prior_H = None   # ndarray or None
        self._prior_b = None   # ndarray or None
        self._prior_kf_ids = []  # which kf_ids the prior references

        # IMU noise for bias-walk prior between consecutive keyframes
        # (used as a soft prior on bias change)
        self._sigma_ba = 3.0e-3
        self._sigma_bg = 1.9393e-5

    # ---- Public API ----------------------------------------------------------

    def add_keyframe(self, state, preintegrator=None):
        """
        Add a new keyframe to the sliding window.

        Parameters
        ----------
        state : KeyframeState
            State of the new keyframe.
        preintegrator : ImuPreintegrator or None
            Preintegrated IMU measurements from the previous keyframe to this one.
            None for the very first keyframe.

        Returns
        -------
        kf_id : int
            Unique identifier assigned to this keyframe.
        """
        kf_id = self._next_kf_id
        self._next_kf_id += 1

        self._keyframes.append(state.copy())
        self._kf_ids.append(kf_id)

        # Add IMU factor connecting previous keyframe to this one
        if preintegrator is not None and len(self._kf_ids) >= 2:
            prev_kf_id = self._kf_ids[-2]
            self._imu_factors[prev_kf_id] = (kf_id, preintegrator)

        # Marginalize oldest keyframe if window exceeds maximum size
        if len(self._keyframes) > self.window_size:
            self._marginalize_oldest()

        return kf_id

    def add_visual_factor(self, kf_id, lm_id, uv_obs, lm_p3d_world=None):
        """
        Register a 2D observation of a landmark from a keyframe.

        Parameters
        ----------
        kf_id : int
            Keyframe identifier (must be in current window).
        lm_id : int
            Landmark identifier.
        uv_obs : array (2,)
            Observed pixel coordinates [u, v].
        lm_p3d_world : array (3,) or None
            Initial 3D position in world frame.  Used to initialize the
            landmark if it has not been seen before.
        """
        if kf_id not in self._kf_ids:
            return  # stale keyframe, skip

        uv = np.array(uv_obs, dtype=np.float64)

        if lm_id not in self._landmarks and lm_p3d_world is not None:
            self._landmarks[lm_id] = LandmarkState(lm_p3d_world, lm_id)

        if lm_id in self._landmarks:
            self._vis_factors.append((kf_id, lm_id, uv))

    def add_pose_factor(self, kf_id, p_meas, q_meas,
                        sigma_trans=0.05, sigma_rot=0.05):
        """
        Add a 6-DOF VO pose measurement factor for a keyframe (Stage A).

        Residual: r = [p_kf - p_meas;  log_so3(R_meas^T @ R_kf)]  in R^6
        This anchors the keyframe near its VO pose estimate while allowing
        the optimizer to refine it for IMU consistency.

        Parameters
        ----------
        kf_id : int
            Keyframe to anchor.
        p_meas : array (3,)
            VO position measurement in world frame.
        q_meas : array (4,)
            VO orientation measurement [x,y,z,w].
        sigma_trans : float
            Translation noise std dev [m].
        sigma_rot : float
            Rotation noise std dev [rad].
        """
        if kf_id not in self._kf_ids:
            return
        self._pose_factors.append((
            kf_id,
            np.array(p_meas, dtype=np.float64),
            np.array(q_meas, dtype=np.float64) / np.linalg.norm(q_meas),
            float(sigma_trans),
            float(sigma_rot)
        ))

    def add_velocity_prior(self, kf_id, v_meas, sigma_v=0.5):
        """
        Add a soft prior on the velocity of a keyframe.

        Residual: r = v_kf - v_meas  in R^3
        Prevents velocity from drifting to arbitrary values in the absence of
        any absolute velocity measurement (velocity is only relatively constrained
        by IMU preintegration factors).

        Parameters
        ----------
        kf_id : int
        v_meas : array (3,)   Expected velocity [m/s] (typically zeros at init).
        sigma_v : float        Velocity noise std dev [m/s].
        """
        if kf_id not in self._kf_ids:
            return
        self._vel_priors.append((
            kf_id,
            np.array(v_meas, dtype=np.float64),
            float(sigma_v)
        ))

    def get_keyframe(self, kf_id):
        """Return the KeyframeState for a given kf_id, or None."""
        try:
            idx = self._kf_ids.index(kf_id)
            return self._keyframes[idx]
        except ValueError:
            return None

    def latest_keyframe(self):
        """Return (kf_id, KeyframeState) of the most recent keyframe."""
        if not self._keyframes:
            return None, None
        return self._kf_ids[-1], self._keyframes[-1]

    def num_keyframes(self):
        """Return the number of keyframes currently in the window."""
        return len(self._keyframes)

    # ---- IMU residual --------------------------------------------------------

    def _imu_residual(self, kf_i, kf_j, preint):
        """
        Compute the 9-dimensional IMU preintegration residual.

        r = [r_p(3), r_v(3), r_R(3)]

        r_p = R_i^T @ (p_j - p_i - v_i * dt - 0.5 * g * dt^2) - delta_p_corr
        r_v = R_i^T @ (v_j - v_i - g * dt) - delta_v_corr
        r_R = Log(delta_R_corr^T @ R_i^T @ R_j)

        Parameters
        ----------
        kf_i : KeyframeState
            State at keyframe i (start of preintegration interval).
        kf_j : KeyframeState
            State at keyframe j (end of preintegration interval).
        preint : ImuPreintegrator
            Preintegrated measurements from i to j.

        Returns
        -------
        r : ndarray (9,)
            Residual vector.
        """
        R_i = quat_to_rot(kf_i.q)
        R_j = quat_to_rot(kf_j.q)
        dt = preint.dt_total
        g = self.gravity

        dp_corr, dv_corr, dR_corr = preint.bias_corrected_measurement(kf_i.b_a, kf_i.b_g)

        # Position residual
        r_p = R_i.T @ (kf_j.p - kf_i.p - kf_i.v * dt - 0.5 * g * dt**2) - dp_corr
        # Velocity residual
        r_v = R_i.T @ (kf_j.v - kf_i.v - g * dt) - dv_corr
        # Rotation residual
        r_R = log_so3(dR_corr.T @ R_i.T @ R_j)

        return np.concatenate([r_p, r_v, r_R])

    def _imu_jacobians_numerical(self, kf_i, kf_j, preint, eps=1e-6):
        """
        Compute IMU factor Jacobians via forward finite differences.

        Returns J_i (9x15) and J_j (9x15) where the columns correspond to
        the 15-dimensional tangent-space perturbation of each keyframe:
        [dp, dv, dtheta, db_a, db_g].

        Parameters
        ----------
        kf_i, kf_j : KeyframeState
        preint : ImuPreintegrator
        eps : float
            Finite-difference step size.

        Returns
        -------
        J_i : ndarray (9, 15)
            Jacobian w.r.t. keyframe i state.
        J_j : ndarray (9, 15)
            Jacobian w.r.t. keyframe j state.
        """
        r0 = self._imu_residual(kf_i, kf_j, preint)

        J_i = np.zeros((9, KF_DIM), dtype=np.float64)
        J_j = np.zeros((9, KF_DIM), dtype=np.float64)

        # Perturb keyframe i
        for k in range(KF_DIM):
            dx = np.zeros(KF_DIM, dtype=np.float64)
            dx[k] = eps
            kf_i_pert = kf_i.copy()
            kf_i_pert.retract(dx)
            r_pert = self._imu_residual(kf_i_pert, kf_j, preint)
            J_i[:, k] = (r_pert - r0) / eps

        # Perturb keyframe j
        for k in range(KF_DIM):
            dx = np.zeros(KF_DIM, dtype=np.float64)
            dx[k] = eps
            kf_j_pert = kf_j.copy()
            kf_j_pert.retract(dx)
            r_pert = self._imu_residual(kf_i, kf_j_pert, preint)
            J_j[:, k] = (r_pert - r0) / eps

        return J_i, J_j

    # ---- Visual residual and Jacobian ----------------------------------------

    def _visual_residual_and_jacobian(self, kf, lm, uv_obs):
        """
        Compute reprojection residual and analytical Jacobians.

        Projects landmark lm.p from world frame into the camera attached to
        keyframe kf, and computes the 2D reprojection error.

        r = uv_obs - [fx * pc[0]/pc[2] + cx,  fy * pc[1]/pc[2] + cy]

        Parameters
        ----------
        kf : KeyframeState
            Keyframe state.
        lm : LandmarkState
            Landmark state.
        uv_obs : ndarray (2,)
            Observed pixel coordinates.

        Returns
        -------
        r : ndarray (2,) or None
            Reprojection residual, or None if landmark is behind camera.
        J_kf : ndarray (2, 6) or None
            Jacobian w.r.t. keyframe [dtheta(3), dp(3)] -- only pose, not velocity/bias.
            Note: we return J w.r.t. the full 15-dim tangent but only pose columns
            are nonzero.  For efficiency, we return (2, 15) with zeros padded.
        J_lm : ndarray (2, 3) or None
            Jacobian w.r.t. landmark position.
        """
        R_wb = quat_to_rot(kf.q)  # body -> world

        # Camera -> world:  p_world = R_wb @ (R_bc @ p_cam + t_bc) + p_body
        # So: world -> camera:
        #   p_cam = R_bc^T @ (R_wb^T @ (p_world - p_body) - t_bc)
        R_bw = R_wb.T
        R_cb = self.R_bc.T  # cam <- body

        # Transform world point to camera frame
        p_b = R_bw @ (lm.p - kf.p)       # landmark in body frame
        p_c = R_cb @ (p_b - self.t_bc)    # landmark in camera frame

        # Depth check: skip if behind camera
        if p_c[2] < 0.1:
            return None, None, None

        # Project to pixel coordinates
        inv_z = 1.0 / p_c[2]
        u_proj = self.fx * p_c[0] * inv_z + self.cx
        v_proj = self.fy * p_c[1] * inv_z + self.cy

        # Residual: observation minus projection
        r = uv_obs - np.array([u_proj, v_proj], dtype=np.float64)

        # ---- Jacobian of projection w.r.t. p_c (2x3) ----
        J_proj = np.array([
            [self.fx * inv_z, 0.0,            -self.fx * p_c[0] * inv_z**2],
            [0.0,            self.fy * inv_z, -self.fy * p_c[1] * inv_z**2]
        ], dtype=np.float64)

        # Negative sign because r = obs - proj, so dr/d(proj) = -I,
        # and d(proj)/d(p_c) = J_proj.  Combined: dr/d(p_c) = -J_proj.
        J_pc = -J_proj

        # ---- Chain rule: d(p_c)/d(state) ----
        # p_c = R_cb @ R_bw @ (lm.p - kf.p) - R_cb @ t_bc
        # Let q_b = R_bw @ (lm.p - kf.p) = p_b  (landmark in body frame)

        # d(p_c)/d(kf.p):
        #   p_b = R_bw @ (lm.p - kf.p)  =>  d(p_b)/d(kf.p) = -R_bw
        #   p_c = R_cb @ p_b - R_cb @ t_bc  =>  d(p_c)/d(p_b) = R_cb
        #   d(p_c)/d(kf.p) = R_cb @ (-R_bw) = -R_cb @ R_bw = -R_cw
        R_cw = R_cb @ R_bw
        dp_c_dp = -R_cw   # (3x3)

        # d(p_c)/d(dtheta):
        #   A rotation perturbation dtheta on kf means R_wb -> R_wb @ Exp(dtheta)
        #   => R_bw -> Exp(-dtheta) @ R_bw
        #   p_b -> Exp(-dtheta) @ R_bw @ (lm.p - kf.p) ~ (I - [dtheta]_x) @ p_b
        #   dp_b/d(dtheta) = -[p_b]_x  => dp_b = [p_b]_x @ dtheta  (sign flip of skew)
        #   Actually: d(Exp(-dtheta) @ p_b)/d(dtheta) = [p_b]_x  (right perturbation)
        #   Wait -- let's be precise.
        #
        #   With right perturbation on SO(3):
        #     R_wb_new = R_wb @ Exp(dtheta)
        #     R_bw_new = Exp(-dtheta) @ R_bw ≈ (I - [dtheta]_x) @ R_bw
        #     p_b_new = R_bw_new @ (lm.p - kf.p) = p_b - [dtheta]_x @ p_b
        #             = p_b + [p_b]_x @ dtheta
        #   So dp_b/d(dtheta) = [p_b]_x = skew(p_b)
        #   dp_c/d(dtheta) = R_cb @ skew(p_b)
        dp_c_dtheta = R_cb @ _skew(p_b)  # (3x3)

        # d(p_c)/d(lm.p):
        #   d(p_c)/d(lm.p) = R_cb @ R_bw = R_cw
        dp_c_dlm = R_cw  # (3x3)

        # ---- Assemble full Jacobians ----
        # J_kf: (2, 15) -- layout: [dp(3), dv(3), dtheta(3), db_a(3), db_g(3)]
        #   Only dp and dtheta columns are nonzero.
        J_kf = np.zeros((2, KF_DIM), dtype=np.float64)
        J_kf[:, 0:3] = J_pc @ dp_c_dp       # d(r)/d(kf.p)
        J_kf[:, 6:9] = J_pc @ dp_c_dtheta   # d(r)/d(kf.theta)

        # J_lm: (2, 3)
        J_lm = J_pc @ dp_c_dlm

        return r, J_kf, J_lm

    # ---- Hessian assembly ----------------------------------------------------

    def _assemble_hessian(self):
        """
        Build the full Gauss-Newton Hessian H and gradient b from all factors.

        State ordering in H:
            [kf_0(15) | kf_1(15) | ... | kf_{N-1}(15) | lm_0(3) | lm_1(3) | ...]

        Returns
        -------
        H : ndarray (n, n)
            Approximate Hessian (J^T Sigma^{-1} J summed over all factors).
        b : ndarray (n,)
            Gradient vector (J^T Sigma^{-1} r summed over all factors).
        cost : float
            Total weighted squared residual (0.5 * r^T Sigma^{-1} r).
        """
        n_kf = len(self._keyframes)

        # Build landmark index mapping: lm_id -> sequential index
        active_lm_ids = sorted(set(lm_id for _, lm_id, _ in self._vis_factors
                                    if lm_id in self._landmarks))
        lm_id_to_idx = {lm_id: i for i, lm_id in enumerate(active_lm_ids)}
        n_lm = len(active_lm_ids)

        total_dim = n_kf * KF_DIM + n_lm * LM_DIM
        H = np.zeros((total_dim, total_dim), dtype=np.float64)
        b = np.zeros(total_dim, dtype=np.float64)
        cost = 0.0

        # ---- Prior factor (from marginalization) ----
        if self._prior_H is not None:
            self._add_prior_to_hessian(H, b)

        # ---- VO pose measurement factors (Stage A absolute anchors) ----
        for kf_id, p_meas, q_meas, sigma_t, sigma_r in self._pose_factors:
            if kf_id not in self._kf_ids:
                continue
            kf_idx = self._kf_ids.index(kf_id)
            kf = self._keyframes[kf_idx]
            off = kf_idx * KF_DIM

            R_kf   = quat_to_rot(kf.q)
            R_meas = quat_to_rot(q_meas)

            # Residual: r_t = p_kf - p_meas,  r_R = log_so3(R_meas^T @ R_kf)
            r_t = kf.p - p_meas
            r_R = log_so3(R_meas.T @ R_kf)
            r_pose = np.concatenate([r_t, r_R])  # (6,)

            # Information matrix (diagonal)
            info_t = 1.0 / (sigma_t ** 2)
            info_r = 1.0 / (sigma_r ** 2)
            info_diag = np.array([info_t]*3 + [info_r]*3)

            # Jacobian: dr_t/dp = I(3),  dr_R/dtheta = I(3)  (at linearization point)
            # Only the [p(0:3), theta(6:9)] blocks of the 15-DOF tangent space
            J_pose = np.zeros((6, KF_DIM), dtype=np.float64)
            J_pose[0:3, 0:3] = np.eye(3)   # dr_t / d(delta_p)
            J_pose[3:6, 6:9] = np.eye(3)   # dr_R / d(delta_theta)

            # H += J^T diag(info) J,  b += J^T diag(info) r
            IJ = info_diag[:, None] * J_pose       # (6, 15)
            H[off:off+KF_DIM, off:off+KF_DIM] += J_pose.T @ IJ
            b[off:off+KF_DIM] += J_pose.T @ (info_diag * r_pose)
            cost += 0.5 * np.dot(info_diag * r_pose, r_pose)

        # ---- Velocity priors ----
        for kf_id, v_meas, sigma_v in self._vel_priors:
            if kf_id not in self._kf_ids:
                continue
            kf_idx = self._kf_ids.index(kf_id)
            kf = self._keyframes[kf_idx]
            off = kf_idx * KF_DIM

            r_v = kf.v - v_meas          # (3,)
            info_v = 1.0 / (sigma_v ** 2)

            # Jacobian dr_v/d(delta_v) = I(3), lives at tangent indices [3:6]
            H[off+3:off+6, off+3:off+6] += info_v * np.eye(3)
            b[off+3:off+6] += info_v * r_v
            cost += 0.5 * info_v * np.dot(r_v, r_v)

        # ---- IMU preintegration factors ----
        for kf_id_i, (kf_id_j, preint) in self._imu_factors.items():
            if kf_id_i not in self._kf_ids or kf_id_j not in self._kf_ids:
                continue
            idx_i = self._kf_ids.index(kf_id_i)
            idx_j = self._kf_ids.index(kf_id_j)
            kf_i = self._keyframes[idx_i]
            kf_j = self._keyframes[idx_j]

            r = self._imu_residual(kf_i, kf_j, preint)
            J_i, J_j = self._imu_jacobians_numerical(kf_i, kf_j, preint)

            # Information matrix: inverse of preintegration covariance.
            # Apply a minimum covariance floor so the IMU factor never
            # dominates the Hessian by more than ~1000× relative to the
            # VO pose factor (info ~400 for sigma_t=5 cm).
            # Floor values correspond to sigma: dp~1cm, dv~0.3m/s, dR~0.5deg.
            _COV_FLOOR = np.diag([1e-4, 1e-4, 1e-4,   # position  (1 cm)
                                   0.09, 0.09, 0.09,    # velocity  (0.3 m/s)
                                   7.6e-5, 7.6e-5, 7.6e-5])  # rotation (0.5 deg)
            Cov = np.maximum(preint.Cov, _COV_FLOOR) + 1e-8 * np.eye(9, dtype=np.float64)
            try:
                Sigma_inv = np.linalg.inv(Cov)
            except np.linalg.LinAlgError:
                Sigma_inv = np.linalg.pinv(Cov)
            # Enforce symmetry
            Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.T)

            # Concatenated Jacobian for this factor: [J_i | J_j]
            # but they map to different state blocks
            off_i = idx_i * KF_DIM
            off_j = idx_j * KF_DIM

            # H += J^T Sigma^{-1} J,  b += J^T Sigma^{-1} r
            SJ_i = Sigma_inv @ J_i  # (9, 15)
            SJ_j = Sigma_inv @ J_j  # (9, 15)

            H[off_i:off_i+KF_DIM, off_i:off_i+KF_DIM] += J_i.T @ SJ_i
            H[off_i:off_i+KF_DIM, off_j:off_j+KF_DIM] += J_i.T @ SJ_j
            H[off_j:off_j+KF_DIM, off_i:off_i+KF_DIM] += J_j.T @ SJ_i
            H[off_j:off_j+KF_DIM, off_j:off_j+KF_DIM] += J_j.T @ SJ_j

            Sr = Sigma_inv @ r
            b[off_i:off_i+KF_DIM] += J_i.T @ Sr
            b[off_j:off_j+KF_DIM] += J_j.T @ Sr

            cost += 0.5 * r @ Sr

            # ---- Bias random-walk factor between consecutive keyframes ----
            # Soft prior: ||b_a_j - b_a_i||^2 / sigma_ba^2 + ||b_g_j - b_g_i||^2 / sigma_bg^2
            # This constrains bias from changing too fast.
            dt = preint.dt_total
            if dt > 0:
                r_ba = kf_j.b_a - kf_i.b_a
                r_bg = kf_j.b_g - kf_i.b_g
                # Information: 1/(sigma^2 * dt)
                info_ba = 1.0 / (self._sigma_ba**2 * dt + 1e-12)
                info_bg = 1.0 / (self._sigma_bg**2 * dt + 1e-12)

                # Jacobian for bias walk: dr/d(b_a_i) = -I, dr/d(b_a_j) = +I
                # Indices in tangent space: b_a = [9:12], b_g = [12:15]
                for dim_offset, r_b, info_b in [(9, r_ba, info_ba), (12, r_bg, info_bg)]:
                    off_i_b = off_i + dim_offset
                    off_j_b = off_j + dim_offset
                    # H contributions (3x3 blocks)
                    H[off_i_b:off_i_b+3, off_i_b:off_i_b+3] += info_b * np.eye(3)
                    H[off_i_b:off_i_b+3, off_j_b:off_j_b+3] -= info_b * np.eye(3)
                    H[off_j_b:off_j_b+3, off_i_b:off_i_b+3] -= info_b * np.eye(3)
                    H[off_j_b:off_j_b+3, off_j_b:off_j_b+3] += info_b * np.eye(3)
                    # b contributions
                    b[off_i_b:off_i_b+3] -= info_b * r_b
                    b[off_j_b:off_j_b+3] += info_b * r_b
                    cost += 0.5 * info_b * np.dot(r_b, r_b)

        # ---- Visual reprojection factors ----
        # Information matrix for visual factors: identity / sigma_px^2
        sigma_px = 1.0  # 1 pixel standard deviation
        info_vis = 1.0 / (sigma_px ** 2)

        for kf_id, lm_id, uv_obs in self._vis_factors:
            if kf_id not in self._kf_ids or lm_id not in self._landmarks:
                continue
            if lm_id not in lm_id_to_idx:
                continue

            kf_idx = self._kf_ids.index(kf_id)
            kf = self._keyframes[kf_idx]
            lm = self._landmarks[lm_id]

            result = self._visual_residual_and_jacobian(kf, lm, uv_obs)
            if result[0] is None:
                continue
            r, J_kf, J_lm = result

            off_kf = kf_idx * KF_DIM
            off_lm = n_kf * KF_DIM + lm_id_to_idx[lm_id] * LM_DIM

            # Weighted Jacobians
            wJ_kf = info_vis * J_kf  # (2, 15)
            wJ_lm = info_vis * J_lm  # (2, 3)
            wr = info_vis * r         # (2,)

            # H blocks
            H[off_kf:off_kf+KF_DIM, off_kf:off_kf+KF_DIM] += J_kf.T @ wJ_kf
            H[off_kf:off_kf+KF_DIM, off_lm:off_lm+LM_DIM] += J_kf.T @ wJ_lm
            H[off_lm:off_lm+LM_DIM, off_kf:off_kf+KF_DIM] += J_lm.T @ wJ_kf
            H[off_lm:off_lm+LM_DIM, off_lm:off_lm+LM_DIM] += J_lm.T @ wJ_lm

            # b
            b[off_kf:off_kf+KF_DIM] += J_kf.T @ wr
            b[off_lm:off_lm+LM_DIM] += J_lm.T @ wr

            cost += 0.5 * r @ wr

        return H, b, cost

    def _add_prior_to_hessian(self, H, b):
        """
        Re-inject the stored marginalization prior into the assembled Hessian.

        The prior references specific keyframe IDs stored in self._prior_kf_ids.
        We map those IDs to their current positions in the state vector and
        add the prior H and b blocks at the correct locations.
        """
        if self._prior_H is None:
            return

        # Map prior kf_ids to current indices
        idx_map = []
        for kf_id in self._prior_kf_ids:
            if kf_id in self._kf_ids:
                idx_map.append(self._kf_ids.index(kf_id))
            else:
                return  # prior references a keyframe no longer in window; skip

        prior_dim = len(self._prior_kf_ids) * KF_DIM
        if self._prior_H.shape[0] != prior_dim:
            return  # dimension mismatch safety check

        for pi, ci in enumerate(idx_map):
            for pj, cj in enumerate(idx_map):
                ri = ci * KF_DIM
                rj = cj * KF_DIM
                pri = pi * KF_DIM
                prj = pj * KF_DIM
                H[ri:ri+KF_DIM, rj:rj+KF_DIM] += self._prior_H[pri:pri+KF_DIM, prj:prj+KF_DIM]

        for pi, ci in enumerate(idx_map):
            ri = ci * KF_DIM
            pri = pi * KF_DIM
            b[ri:ri+KF_DIM] += self._prior_b[pri:pri+KF_DIM]

    # ---- Optimization --------------------------------------------------------

    def optimize(self, max_iters=5, lm_lambda_init=1e-4):
        """
        Run Levenberg-Marquardt optimization on the sliding window.

        Iteratively linearizes all factors, assembles the Gauss-Newton Hessian,
        applies diagonal damping (LM), solves for the state update, and
        applies it via manifold retraction.

        Parameters
        ----------
        max_iters : int
            Maximum number of LM iterations.
        lm_lambda_init : float
            Initial LM damping parameter.

        Returns
        -------
        cost : float
            Final cost after optimization.
        """
        if len(self._keyframes) < 2:
            return 0.0

        lm_lambda = lm_lambda_init
        LM_LAMBDA_MIN = 1e-8
        LM_LAMBDA_MAX = 1e8

        # Build landmark index (needed for retraction)
        active_lm_ids = sorted(set(lm_id for _, lm_id, _ in self._vis_factors
                                    if lm_id in self._landmarks))

        for iteration in range(max_iters):
            H, b_vec, cost = self._assemble_hessian()
            n = H.shape[0]

            if n == 0:
                return 0.0

            # LM damping: H_damp = H + lambda * diag(H)
            diag_H = np.diag(H).copy()
            diag_H = np.maximum(diag_H, 1e-6)  # avoid zero diagonal
            H_damp = H + lm_lambda * np.diag(diag_H)

            # Add small regularization for numerical stability
            H_damp += 1e-8 * np.eye(n, dtype=np.float64)

            # Solve H_damp @ dx = -b  (note: b already contains J^T Sigma^{-1} r)
            try:
                dx = np.linalg.solve(H_damp, -b_vec)
            except np.linalg.LinAlgError:
                lm_lambda = min(lm_lambda * 10.0, LM_LAMBDA_MAX)
                continue

            # Check for NaN/Inf in solution
            if not np.all(np.isfinite(dx)):
                lm_lambda = min(lm_lambda * 10.0, LM_LAMBDA_MAX)
                continue

            # Save current state for potential rollback
            kf_backup = [kf.copy() for kf in self._keyframes]
            lm_backup = {lid: lm.copy() for lid, lm in self._landmarks.items()}

            # Apply retraction to keyframes
            n_kf = len(self._keyframes)
            for i in range(n_kf):
                off = i * KF_DIM
                self._keyframes[i].retract(dx[off:off+KF_DIM])

            # Apply retraction to landmarks
            for li, lm_id in enumerate(active_lm_ids):
                off = n_kf * KF_DIM + li * LM_DIM
                if lm_id in self._landmarks:
                    self._landmarks[lm_id].p += dx[off:off+LM_DIM]

            # Evaluate new cost
            _, _, new_cost = self._assemble_hessian()

            if new_cost < cost:
                # Accept step, decrease lambda
                lm_lambda = max(lm_lambda / 3.0, LM_LAMBDA_MIN)
            else:
                # Reject step, restore state, increase lambda
                self._keyframes = kf_backup
                for lid, lm in lm_backup.items():
                    self._landmarks[lid] = lm
                lm_lambda = min(lm_lambda * 10.0, LM_LAMBDA_MAX)

        # Return final cost
        _, _, final_cost = self._assemble_hessian()
        return final_cost

    # ---- Marginalization -----------------------------------------------------

    def _marginalize_oldest(self):
        """
        Marginalize the oldest keyframe using Schur complement.

        Steps:
        1. Identify landmarks observed ONLY from the oldest keyframe.
        2. Assemble the full Hessian.
        3. Partition into alpha (oldest kf + exclusive landmarks) and
           beta (remaining keyframes + shared landmarks).
        4. Compute the Schur complement: H_prior = H_bb - H_ba H_aa^{-1} H_ab.
        5. Store the prior and remove the oldest keyframe.
        """
        if len(self._keyframes) < 2:
            return

        oldest_kf_id = self._kf_ids[0]

        # Find landmarks observed ONLY from the oldest keyframe
        # (these will be marginalized out too)
        lm_observers = {}  # lm_id -> set of kf_ids that observe it
        for kf_id, lm_id, _ in self._vis_factors:
            if kf_id in self._kf_ids and lm_id in self._landmarks:
                if lm_id not in lm_observers:
                    lm_observers[lm_id] = set()
                lm_observers[lm_id].add(kf_id)

        exclusive_lm_ids = set()
        for lm_id, observers in lm_observers.items():
            if observers == {oldest_kf_id}:
                exclusive_lm_ids.add(lm_id)

        # Assemble the full Hessian before removing anything
        H_full, b_full, _ = self._assemble_hessian()
        n_kf = len(self._keyframes)

        # Build landmark index consistent with _assemble_hessian
        active_lm_ids = sorted(set(lm_id for _, lm_id, _ in self._vis_factors
                                    if lm_id in self._landmarks))
        lm_id_to_idx = {lm_id: i for i, lm_id in enumerate(active_lm_ids)}

        # Determine alpha indices (to marginalize) and beta indices (to keep)
        total_dim = H_full.shape[0]
        alpha_indices = []  # flat indices into H

        # Oldest keyframe
        alpha_indices.extend(range(0, KF_DIM))

        # Exclusive landmarks
        for lm_id in sorted(exclusive_lm_ids):
            if lm_id in lm_id_to_idx:
                off = n_kf * KF_DIM + lm_id_to_idx[lm_id] * LM_DIM
                alpha_indices.extend(range(off, off + LM_DIM))

        beta_indices = [i for i in range(total_dim) if i not in set(alpha_indices)]

        if len(beta_indices) == 0:
            # Nothing to keep -- clear prior
            self._prior_H = None
            self._prior_b = None
            self._prior_kf_ids = []
        else:
            alpha_idx = np.array(alpha_indices, dtype=int)
            beta_idx = np.array(beta_indices, dtype=int)

            H_aa = H_full[np.ix_(alpha_idx, alpha_idx)]
            H_ab = H_full[np.ix_(alpha_idx, beta_idx)]
            H_ba = H_full[np.ix_(beta_idx, alpha_idx)]
            H_bb = H_full[np.ix_(beta_idx, beta_idx)]
            b_a = b_full[alpha_idx]
            b_b = b_full[beta_idx]

            # Regularize H_aa before inversion
            H_aa += 1e-8 * np.eye(H_aa.shape[0], dtype=np.float64)

            try:
                H_aa_inv = np.linalg.inv(H_aa)
            except np.linalg.LinAlgError:
                H_aa_inv = np.linalg.pinv(H_aa)

            # Schur complement
            H_prior_full = H_bb - H_ba @ H_aa_inv @ H_ab
            b_prior_full = b_b - H_ba @ H_aa_inv @ b_a

            # Enforce symmetry
            H_prior_full = 0.5 * (H_prior_full + H_prior_full.T)

            # The beta block contains keyframes [1:] and shared landmarks.
            # For the prior, we only keep the keyframe part because landmarks
            # may be removed later.  Extract just the keyframe portion.
            remaining_kf_dim = (n_kf - 1) * KF_DIM
            # The first remaining_kf_dim entries in beta correspond to kf[1:]
            # (since kf[0] was in alpha and the remaining kfs come first in beta)

            # Map beta indices back: beta starts at KF_DIM for keyframe states
            # We need to figure out which beta indices are keyframe indices.
            # Beta keyframe indices: positions [KF_DIM, 2*KF_DIM, ..., (n_kf-1)*KF_DIM + 14]
            beta_kf_indices_in_beta = []
            for i, bi in enumerate(beta_indices):
                if bi < n_kf * KF_DIM:
                    beta_kf_indices_in_beta.append(i)

            if len(beta_kf_indices_in_beta) > 0:
                bki = np.array(beta_kf_indices_in_beta, dtype=int)
                H_prior_kf = H_prior_full[np.ix_(bki, bki)]
                b_prior_kf = b_prior_full[bki]

                # Combine with existing prior (if any, from previous marginalizations)
                # The new prior replaces the old one since we already included
                # the old prior in _assemble_hessian.
                self._prior_H = H_prior_kf.copy()
                self._prior_b = b_prior_kf.copy()
                self._prior_kf_ids = self._kf_ids[1:]  # remaining keyframes
            else:
                self._prior_H = None
                self._prior_b = None
                self._prior_kf_ids = []

        # ---- Remove oldest keyframe and associated data ----
        self._keyframes.pop(0)
        self._kf_ids.pop(0)

        # Remove IMU factor from the oldest
        if oldest_kf_id in self._imu_factors:
            del self._imu_factors[oldest_kf_id]

        # Remove visual factors referencing the oldest keyframe
        self._vis_factors = [
            (kf_id, lm_id, uv)
            for kf_id, lm_id, uv in self._vis_factors
            if kf_id != oldest_kf_id
        ]

        # Remove exclusive landmarks
        for lm_id in exclusive_lm_ids:
            if lm_id in self._landmarks:
                del self._landmarks[lm_id]

        # Remove visual factors referencing deleted landmarks
        self._vis_factors = [
            (kf_id, lm_id, uv)
            for kf_id, lm_id, uv in self._vis_factors
            if lm_id in self._landmarks
        ]

        # Remove pose and velocity factors referencing the oldest keyframe
        self._pose_factors = [
            f for f in self._pose_factors if f[0] != oldest_kf_id
        ]
        self._vel_priors = [
            f for f in self._vel_priors if f[0] != oldest_kf_id
        ]

        # Sanity check: no stale kf_id references
        valid_kf_set = set(self._kf_ids)
        self._vis_factors = [
            (kf_id, lm_id, uv)
            for kf_id, lm_id, uv in self._vis_factors
            if kf_id in valid_kf_set
        ]
