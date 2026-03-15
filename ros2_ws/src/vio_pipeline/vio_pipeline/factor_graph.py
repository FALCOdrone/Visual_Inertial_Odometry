"""
factor_graph.py
===============
Sliding-window factor graph with IMU preintegration factors, VO relative-pose
factors, and Schur-complement marginalization.

Pose-level graph: no landmark variables. Each keyframe node has 15 DOF:
  [delta_theta(3), delta_p(3), delta_v(3), delta_b_a(3), delta_b_g(3)]

Optimized via Levenberg-Marquardt on a dense system (150x150 for W=10).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from vio_pipeline.so3_utils import (
    skew, exp_so3, log_so3, right_jacobian_so3, inv_right_jacobian_so3,
)
from vio_pipeline.imu_preintegrator import ImuPreintegrator


# ── Keyframe state ────────────────────────────────────────────────────────────


@dataclass
class KeyframeState:
    """State of a single keyframe node in the factor graph."""
    stamp_ns: int
    R: np.ndarray        # 3x3 rotation (world <- body)
    p: np.ndarray        # (3,) position in world
    v: np.ndarray        # (3,) velocity in world
    b_a: np.ndarray      # (3,) accelerometer bias
    b_g: np.ndarray      # (3,) gyroscope bias

    def copy(self) -> KeyframeState:
        return KeyframeState(
            stamp_ns=self.stamp_ns,
            R=self.R.copy(),
            p=self.p.copy(),
            v=self.v.copy(),
            b_a=self.b_a.copy(),
            b_g=self.b_g.copy(),
        )

    def retract(self, delta: np.ndarray) -> KeyframeState:
        """Manifold oplus: apply 15-dim tangent vector [dtheta, dp, dv, dba, dbg]."""
        return KeyframeState(
            stamp_ns=self.stamp_ns,
            R=self.R @ exp_so3(delta[0:3]),
            p=self.p + delta[3:6],
            v=self.v + delta[6:9],
            b_a=self.b_a + delta[9:12],
            b_g=self.b_g + delta[12:15],
        )

    def local(self, other: KeyframeState) -> np.ndarray:
        """Manifold ominus: this (-) other -> 15-dim tangent vector."""
        delta = np.zeros(15, dtype=np.float64)
        delta[0:3] = log_so3(self.R.T @ other.R)
        delta[3:6] = other.p - self.p
        delta[6:9] = other.v - self.v
        delta[9:12] = other.b_a - self.b_a
        delta[12:15] = other.b_g - self.b_g
        return delta


# ── Factor types ──────────────────────────────────────────────────────────────


@dataclass
class ImuFactor:
    """IMU preintegration factor connecting two consecutive keyframe nodes."""
    i: int  # index of first keyframe
    j: int  # index of second keyframe
    preint: ImuPreintegrator
    # Information for bias random walk (filled at construction)
    info_ba: np.ndarray = field(default_factory=lambda: np.eye(3))
    info_bg: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        dt = max(self.preint.delta_t, 1e-6)
        self.info_ba = np.eye(3) / (self.preint._sigma_ba**2 * dt)
        self.info_bg = np.eye(3) / (self.preint._sigma_bg**2 * dt)


@dataclass
class VoFactor:
    """Visual odometry relative-pose factor connecting two keyframe nodes."""
    i: int
    j: int
    dR: np.ndarray       # 3x3 relative rotation measurement
    dp: np.ndarray       # (3,) relative position measurement
    info: np.ndarray     # 6x6 information matrix


@dataclass
class PriorFactor:
    """Linearized prior from marginalization (multi-node)."""
    node_indices: list[int]        # which nodes this prior constrains
    J: np.ndarray                  # (d x len(node_indices)*15) Jacobian
    r0: np.ndarray                 # (d,) residual at linearization point
    x_lins: list[KeyframeState]    # linearization points for each node


# ── Sliding window graph ─────────────────────────────────────────────────────


class SlidingWindowGraph:
    """
    Sliding-window factor graph with LM optimization and Schur marginalization.
    """

    def __init__(
        self,
        window_size: int = 10,
        gravity: np.ndarray | None = None,
        lm_max_iter: int = 5,
        lm_lambda_init: float = 1e-3,
    ):
        self._window_size = window_size
        self._gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])
        self._lm_max_iter = lm_max_iter
        self._lm_lambda_init = lm_lambda_init

        self._states: list[KeyframeState] = []
        self._imu_factors: list[ImuFactor] = []
        self._vo_factors: list[VoFactor] = []
        self._prior: Optional[PriorFactor] = None

    @property
    def num_keyframes(self) -> int:
        return len(self._states)

    def add_keyframe(self, state: KeyframeState) -> int:
        """Add a new keyframe to the window. Returns its index."""
        self._states.append(state.copy())
        return len(self._states) - 1

    def add_imu_factor(self, i: int, j: int, preint: ImuPreintegrator) -> None:
        self._imu_factors.append(ImuFactor(i=i, j=j, preint=preint))

    def add_vo_factor(self, i: int, j: int, dR: np.ndarray, dp: np.ndarray, cov_6x6: np.ndarray) -> None:
        info = np.zeros((6, 6), dtype=np.float64)
        diag = np.diag(cov_6x6)
        for k in range(6):
            info[k, k] = 1.0 / max(diag[k], 1e-12)
        self._vo_factors.append(VoFactor(i=i, j=j, dR=dR, dp=dp, info=info))

    def latest_state(self) -> KeyframeState:
        return self._states[-1].copy()

    def latest_biases(self):
        s = self._states[-1]
        return s.b_a.copy(), s.b_g.copy()

    # ── Build linear system ──────────────────────────────────────────────────

    def _build_system(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Build the normal equations H*dx = -b and return (H, b, cost)."""
        n = len(self._states)
        dim = 15 * n
        H = np.zeros((dim, dim), dtype=np.float64)
        b = np.zeros(dim, dtype=np.float64)
        cost = 0.0

        # IMU factors
        for fac in self._imu_factors:
            r, Ji, Jj, omega = self._imu_residual_and_jacobians(fac)
            cost += r @ omega @ r
            si, sj = fac.i * 15, fac.j * 15
            # H += J^T * Omega * J
            JiTO = Ji.T @ omega
            JjTO = Jj.T @ omega
            H[si:si+15, si:si+15] += JiTO @ Ji
            H[si:si+15, sj:sj+15] += JiTO @ Jj
            H[sj:sj+15, si:si+15] += JjTO @ Ji
            H[sj:sj+15, sj:sj+15] += JjTO @ Jj
            b[si:si+15] += JiTO @ r
            b[sj:sj+15] += JjTO @ r

        # VO factors
        for fac in self._vo_factors:
            r, Ji, Jj = self._vo_residual_and_jacobians(fac)
            omega = fac.info
            cost += r @ omega @ r
            si, sj = fac.i * 15, fac.j * 15
            JiTO = Ji.T @ omega
            JjTO = Jj.T @ omega
            H[si:si+15, si:si+15] += JiTO @ Ji
            H[si:si+15, sj:sj+15] += JiTO @ Jj
            H[sj:sj+15, si:si+15] += JjTO @ Ji
            H[sj:sj+15, sj:sj+15] += JjTO @ Jj
            b[si:si+15] += JiTO @ r
            b[sj:sj+15] += JjTO @ r

        # Prior factor (multi-node)
        if self._prior is not None:
            p = self._prior
            num_prior_nodes = len(p.node_indices)
            dx_full = np.zeros(num_prior_nodes * 15, dtype=np.float64)
            for k, (ni, xl) in enumerate(zip(p.node_indices, p.x_lins)):
                dx_full[k*15:(k+1)*15] = xl.local(self._states[ni])
            r = p.J @ dx_full + p.r0
            cost += float(r @ r)
            for ki, ni in enumerate(p.node_indices):
                idx_i = ni * 15
                Ji = p.J[:, ki*15:(ki+1)*15]
                b[idx_i:idx_i+15] += Ji.T @ r
                for kj, nj in enumerate(p.node_indices):
                    idx_j = nj * 15
                    Jj = p.J[:, kj*15:(kj+1)*15]
                    H[idx_i:idx_i+15, idx_j:idx_j+15] += Ji.T @ Jj

        return H, b, cost

    def _imu_residual_and_jacobians(self, fac: ImuFactor):
        """Compute 15-dim IMU residual and Jacobians w.r.t. states i and j."""
        si = self._states[fac.i]
        sj = self._states[fac.j]
        preint = fac.preint
        dt = preint.delta_t
        g = self._gravity

        # Bias correction
        dR_corr, dv_corr, dp_corr = preint.correct(si.b_a, si.b_g)

        Ri = si.R
        Rj = sj.R

        # Residuals
        r = np.zeros(15, dtype=np.float64)
        # Rotation residual
        r[0:3] = log_so3(dR_corr.T @ Ri.T @ Rj)
        # Velocity residual
        r[3:6] = Ri.T @ (sj.v - si.v - g * dt) - dv_corr
        # Position residual
        r[6:9] = Ri.T @ (sj.p - si.p - si.v * dt - 0.5 * g * dt * dt) - dp_corr
        # Bias residuals
        r[9:12] = sj.b_a - si.b_a
        r[12:15] = sj.b_g - si.b_g

        # Build information matrix (block diagonal)
        # Preintegration covariance for rotation/velocity/position (9x9)
        cov_preint = preint.covariance.copy()
        # Clamp minimum eigenvalues for numerical stability
        eigvals = np.linalg.eigvalsh(cov_preint)
        min_eig = max(np.min(eigvals), 1e-12)
        if min_eig < 1e-12:
            cov_preint += np.eye(9) * 1e-12

        omega = np.zeros((15, 15), dtype=np.float64)
        try:
            omega[0:9, 0:9] = np.linalg.inv(cov_preint)
        except np.linalg.LinAlgError:
            omega[0:9, 0:9] = np.eye(9) * 1e6
        omega[9:12, 9:12] = fac.info_ba
        omega[12:15, 12:15] = fac.info_bg

        # Jacobians (first-order, evaluated at current linearization)
        # Ji: d(residual)/d(delta_xi), Jj: d(residual)/d(delta_xj)
        Ji = np.zeros((15, 15), dtype=np.float64)
        Jj = np.zeros((15, 15), dtype=np.float64)

        # Rotation residual Jacobians
        r_R = log_so3(dR_corr.T @ Ri.T @ Rj)
        Jr_inv = inv_right_jacobian_so3(r_R)
        Ji[0:3, 0:3] = -Jr_inv @ Rj.T @ Ri  # d r_R / d theta_i
        Jj[0:3, 0:3] = Jr_inv                 # d r_R / d theta_j

        # Velocity residual Jacobians
        Ji[3:6, 0:3] = skew(Ri.T @ (sj.v - si.v - g * dt))  # d r_v / d theta_i
        Ji[3:6, 6:9] = -Ri.T                                  # d r_v / d v_i
        Jj[3:6, 6:9] = Ri.T                                   # d r_v / d v_j

        # Position residual Jacobians
        Ji[6:9, 0:3] = skew(Ri.T @ (sj.p - si.p - si.v * dt - 0.5 * g * dt * dt))  # d r_p / d theta_i
        Ji[6:9, 3:6] = -Ri.T                                     # d r_p / d p_i
        Ji[6:9, 6:9] = -Ri.T * dt                              # d r_p / d v_i
        Jj[6:9, 3:6] = Ri.T                                    # d r_p / d p_j

        # Navigation-to-bias Jacobians (from preintegration correction terms)
        # d(r_R)/d(b_g_i): rotation correction depends on gyro bias
        Ji[0:3, 12:15] = -Jr_inv @ Rj.T @ Ri @ dR_corr @ preint.d_R_d_bg

        # d(r_v)/d(b_a_i) and d(r_v)/d(b_g_i)
        Ji[3:6, 9:12]  = -preint.d_v_d_ba
        Ji[3:6, 12:15] = -preint.d_v_d_bg

        # d(r_p)/d(b_a_i) and d(r_p)/d(b_g_i)
        Ji[6:9, 9:12]  = -preint.d_p_d_ba
        Ji[6:9, 12:15] = -preint.d_p_d_bg

        # Bias residual Jacobians
        Ji[9:12, 9:12] = -np.eye(3)   # d r_ba / d ba_i
        Jj[9:12, 9:12] = np.eye(3)    # d r_ba / d ba_j
        Ji[12:15, 12:15] = -np.eye(3)  # d r_bg / d bg_i
        Jj[12:15, 12:15] = np.eye(3)   # d r_bg / d bg_j

        return r, Ji, Jj, omega

    def _vo_residual_and_jacobians(self, fac: VoFactor):
        """Compute 6-dim VO residual and Jacobians."""
        si = self._states[fac.i]
        sj = self._states[fac.j]

        dR_meas = fac.dR
        dp_meas = fac.dp

        # Residuals
        r = np.zeros(6, dtype=np.float64)
        r[0:3] = log_so3(dR_meas.T @ si.R.T @ sj.R)
        r[3:6] = dR_meas.T @ (si.R.T @ (sj.p - si.p) - dp_meas)

        # Jacobians
        Ji = np.zeros((6, 15), dtype=np.float64)
        Jj = np.zeros((6, 15), dtype=np.float64)

        r_R = log_so3(dR_meas.T @ si.R.T @ sj.R)
        Jr_inv = inv_right_jacobian_so3(r_R)

        # Rotation
        Ji[0:3, 0:3] = -Jr_inv @ sj.R.T @ si.R
        Jj[0:3, 0:3] = Jr_inv

        # Position
        dp_body = si.R.T @ (sj.p - si.p)
        Ji[3:6, 0:3] = dR_meas.T @ skew(dp_body)
        Ji[3:6, 3:6] = -dR_meas.T @ si.R.T                      # d r_p / d p_i
        Jj[3:6, 3:6] = dR_meas.T @ si.R.T                       # d r_p / d p_j

        return r, Ji, Jj

    # ── LM Optimizer ─────────────────────────────────────────────────────────

    def _compute_cost(self, states: list[KeyframeState]) -> float:
        """Compute total cost at given state values."""
        old_states = self._states
        self._states = states
        _, _, cost = self._build_system()
        self._states = old_states
        return cost

    def optimize(self) -> float:
        """Run Levenberg-Marquardt optimization. Returns final cost."""
        if len(self._states) < 2:
            return 0.0

        lam = self._lm_lambda_init
        # Per-node step limits: rotation (rad), position (m), velocity (m/s), biases
        max_step = np.array([0.1]*3 + [1.0]*3 + [2.0]*3 + [0.01]*3 + [0.001]*3)

        for iteration in range(self._lm_max_iter):
            H, b_vec, cost = self._build_system()

            # Damping
            H_lm = H + lam * np.diag(np.maximum(np.diag(H), 1e-6))

            try:
                delta = np.linalg.solve(H_lm, -b_vec)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            # Clamp per-node step to prevent divergence
            for k in range(len(self._states)):
                d = delta[k*15:(k+1)*15]
                scale = 1.0
                for i in range(15):
                    if abs(d[i]) > max_step[i] and max_step[i] > 0:
                        scale = min(scale, max_step[i] / abs(d[i]))
                delta[k*15:(k+1)*15] = d * scale

            # Check convergence
            if np.linalg.norm(delta) < 1e-6:
                break

            # Trial update
            trial_states = []
            for k, s in enumerate(self._states):
                trial_states.append(s.retract(delta[k*15:(k+1)*15]))

            trial_cost = self._compute_cost(trial_states)

            if trial_cost < cost:
                self._states = trial_states
                lam = max(lam * 0.5, 1e-8)
            else:
                lam = min(lam * 5.0, 1e6)

        # Return final cost
        _, _, final_cost = self._build_system()
        return final_cost

    # ── Marginalization ──────────────────────────────────────────────────────

    def marginalize_oldest(self) -> None:
        """Remove the oldest keyframe via Schur complement marginalization."""
        if len(self._states) < 2:
            return

        # Build full system at current linearization
        H, b_vec, _ = self._build_system()

        # Partition: node 0 is marginalized, rest are retained
        m = 15  # dims to marginalize
        H_mm = H[0:m, 0:m]
        H_mr = H[0:m, m:]
        H_rm = H[m:, 0:m]
        H_rr = H[m:, m:]
        b_m = b_vec[0:m]
        b_r = b_vec[m:]

        # Schur complement
        try:
            H_mm_inv = np.linalg.inv(H_mm + np.eye(m) * 1e-8)
        except np.linalg.LinAlgError:
            H_mm_inv = np.linalg.pinv(H_mm)

        H_prior_full = H_rr - H_rm @ H_mm_inv @ H_mr
        b_prior_full = b_r - H_rm @ H_mm_inv @ b_m

        # Enforce symmetry
        H_prior_full = (H_prior_full + H_prior_full.T) * 0.5

        # Factor full H_prior = J^T J via eigendecomposition
        dim_retained = H_prior_full.shape[0]
        eigvals, eigvecs = np.linalg.eigh(H_prior_full)
        eigvals = np.maximum(eigvals, 1e-8)
        J_prior = np.diag(np.sqrt(eigvals)) @ eigvecs.T  # (dim_retained x dim_retained)

        # r0 such that J^T r0 = b_prior_full
        try:
            r0 = np.linalg.solve(J_prior.T, b_prior_full)
        except np.linalg.LinAlgError:
            r0 = np.zeros(dim_retained, dtype=np.float64)

        # Save linearization points for all retained nodes (indices 1..n-1)
        num_retained = len(self._states) - 1
        x_lins = [self._states[k + 1].copy() for k in range(num_retained)]

        # Remove oldest state and re-index factors
        self._states.pop(0)

        new_imu = []
        for f in self._imu_factors:
            if f.i > 0:
                new_imu.append(ImuFactor(i=f.i - 1, j=f.j - 1, preint=f.preint))
        self._imu_factors = new_imu

        new_vo = []
        for f in self._vo_factors:
            if f.i > 0:
                new_vo.append(VoFactor(i=f.i - 1, j=f.j - 1, dR=f.dR, dp=f.dp, info=f.info))
        self._vo_factors = new_vo

        # Set full multi-node prior on all retained nodes (indices 0..n-2)
        self._prior = PriorFactor(
            node_indices=list(range(num_retained)),
            J=J_prior,
            r0=r0,
            x_lins=x_lins,
        )
