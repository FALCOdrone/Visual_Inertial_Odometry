#!/usr/bin/env python3
"""
imu_preintegrator.py
====================
On-manifold IMU preintegration following Forster et al., "On-Manifold
Preintegration for Real-Time Visual-Inertial Odometry," TRO 2017.

This module is pure NumPy — no ROS dependency — so it can be unit-tested
independently.

Public API
----------
Module-level SO(3) helpers:
  _skew(v)                 3-vector → 3×3 skew-symmetric matrix
  exp_so3(phi)             rotation vector → 3×3 SO(3) rotation matrix
  log_so3(R)               3×3 SO(3) rotation matrix → rotation vector
  right_jacobian_so3(phi)  right Jacobian Jr of SO(3)

Quaternion helpers (Hamilton [x, y, z, w] convention):
  quat_to_rot(q)           [x,y,z,w] → 3×3 rotation matrix
  rot_to_quat(R)           3×3 → [x,y,z,w] (Shepperd method)
  quat_mul(q1, q2)         Hamilton product q1 ⊗ q2

Main class:
  ImuPreintegrator         Incremental preintegration with bias Jacobians,
                           covariance propagation, first-order bias correction,
                           and full re-integration.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  SO(3) helpers
# ──────────────────────────────────────────────────────────────────────────────


def _skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric (cross-product) matrix.

    For v = [v0, v1, v2]:
        K = [[  0, -v2,  v1],
             [ v2,   0, -v0],
             [-v1,  v0,   0]]
    """
    v = np.asarray(v, dtype=np.float64)
    return np.array(
        [
            [0.0,   -v[2],  v[1]],
            [v[2],   0.0,  -v[0]],
            [-v[1],  v[0],  0.0],
        ],
        dtype=np.float64,
    )


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """Rotation vector → 3×3 SO(3) rotation matrix (Rodrigues formula).

    For theta = ||phi|| < 1e-10: R ≈ I + [phi]_x  (first-order approximation)
    Otherwise:
        K = [phi]_x,   theta = ||phi||
        R = I + sin(theta)/theta * K + (1 - cos(theta))/theta^2 * K^2
    """
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.linalg.norm(phi)
    K = _skew(phi)
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64) + K
    return (
        np.eye(3, dtype=np.float64)
        + (np.sin(theta) / theta) * K
        + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K)
    )


def log_so3(R: np.ndarray) -> np.ndarray:
    """3×3 SO(3) rotation matrix → rotation vector phi.

    cos_theta = (trace(R) - 1) / 2
    theta     = arccos(cos_theta)
    phi       = theta / (2 * sin(theta)) * [R21-R12, R02-R20, R10-R01]

    For theta < 1e-10: returns zero vector (avoid division by zero).
    """
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return np.zeros(3, dtype=np.float64)
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


def right_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian Jr of SO(3) at rotation vector phi.

    Jr = I - (1 - cos(theta)) / theta^2 * [phi]_x
           + (theta - sin(theta)) / theta^3 * [phi]_x^2

    For theta < 1e-8 (small angle): Jr ≈ I - 0.5 * [phi]_x
    """
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.linalg.norm(phi)
    K = _skew(phi)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64) - 0.5 * K
    theta2 = theta * theta
    theta3 = theta2 * theta
    return (
        np.eye(3, dtype=np.float64)
        - ((1.0 - np.cos(theta)) / theta2) * K
        + ((theta - np.sin(theta)) / theta3) * (K @ K)
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Quaternion helpers — Hamilton convention [x, y, z, w]
# ──────────────────────────────────────────────────────────────────────────────


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] → 3×3 rotation matrix (body → world).

    Uses the standard formula:
        R = (w^2 - |v|^2) I + 2 v v^T + 2 w [v]_x
    where v = [x, y, z].
    """
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z),  2.0 * (x * y - w * z),  2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z),  1.0 - 2.0 * (x * x + z * z),  2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y),  2.0 * (y * z + w * x),  1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [x, y, z, w].

    Uses Shepperd's method with 4 branches to avoid numerical issues near
    axes where the denominator would be small.  Output is always normalized.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
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
    q = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, both in [x, y, z, w] convention.

    q = q1 ⊗ q2 represents the composition: first rotate by q2, then by q1.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  ImuPreintegrator
# ──────────────────────────────────────────────────────────────────────────────


class ImuPreintegrator:
    """Incremental IMU preintegration on SO(3) manifold.

    Tracks the preintegrated measurement:
        delta_p  — relative position in frame of keyframe i
        delta_v  — relative velocity in frame of keyframe i
        delta_R  — relative rotation R_i^T @ R_j

    Also tracks the 9×9 covariance of [delta_p, delta_v, delta_theta] and
    the bias Jacobians required for first-order bias correction and for
    building the information matrix in the factor graph.

    Parameters
    ----------
    b_a : array-like (3,)    Accelerometer bias at linearization point [m/s^2]
    b_g : array-like (3,)    Gyroscope bias at linearization point [rad/s]
    sigma_a : float          Accel noise density [m/s^2/sqrt(Hz)]
    sigma_g : float          Gyro noise density  [rad/s/sqrt(Hz)]
    sigma_ba : float         Accel bias random walk [m/s^3/sqrt(Hz)]
    sigma_bg : float         Gyro bias random walk  [rad/s^2/sqrt(Hz)]
    """

    def __init__(
        self,
        b_a,
        b_g,
        sigma_a: float,
        sigma_g: float,
        sigma_ba: float,
        sigma_bg: float,
    ) -> None:
        # ── Preintegrated state ────────────────────────────────────────────
        self.delta_p: np.ndarray = np.zeros(3, dtype=np.float64)
        self.delta_v: np.ndarray = np.zeros(3, dtype=np.float64)
        self.delta_R: np.ndarray = np.eye(3, dtype=np.float64)

        # 9×9 covariance of [delta_p(3), delta_v(3), delta_theta(3)]
        self.Cov: np.ndarray = np.zeros((9, 9), dtype=np.float64)

        # ── Bias Jacobians (3×3 each) ──────────────────────────────────────
        # d(delta_p) / d(b_a), d(delta_p) / d(b_g)
        self.J_p_ba: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        self.J_p_bg: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        # d(delta_v) / d(b_a), d(delta_v) / d(b_g)
        self.J_v_ba: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        self.J_v_bg: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        # d(delta_theta) / d(b_g)  — Forster Eq. 47
        self.J_R_bg: np.ndarray = np.zeros((3, 3), dtype=np.float64)

        # Total integrated time
        self.dt_total: float = 0.0

        # ── Continuous-time noise PSD (6×6) ───────────────────────────────
        # Block diagonal: [sigma_a^2 * I3, sigma_g^2 * I3]
        # This is Q_c in Forster notation.
        self._Q_c: np.ndarray = np.diag(
            np.array([sigma_a**2] * 3 + [sigma_g**2] * 3, dtype=np.float64)
        )

        # ── Store for reintegration ────────────────────────────────────────
        self._sigma_a = float(sigma_a)
        self._sigma_g = float(sigma_g)
        self._sigma_ba = float(sigma_ba)
        self._sigma_bg = float(sigma_bg)
        self.b_a: np.ndarray = np.array(b_a, dtype=np.float64)
        self.b_g: np.ndarray = np.array(b_g, dtype=np.float64)
        # Raw buffer stores (omega_raw, accel_raw, dt) tuples for re-integration
        self._raw_buffer: list = []

    def integrate(
        self,
        omega_corr: np.ndarray,
        accel_corr: np.ndarray,
        dt: float,
        omega_raw: np.ndarray = None,
        accel_raw: np.ndarray = None,
    ) -> None:
        """Integrate one bias-corrected IMU sample.

        CRITICAL: Bias Jacobians are updated BEFORE the state to use the
        pre-step delta_R, as required by the chain rule.  Then the state
        is updated.  Then covariance is propagated using the post-step delta_R.

        Parameters
        ----------
        omega_corr : (3,)  Bias-corrected angular velocity [rad/s]
        accel_corr : (3,)  Bias-corrected specific force   [m/s^2]
        dt         : float Integration timestep            [s]
        omega_raw  : (3,)  Raw angular velocity (for buffer, optional)
        accel_raw  : (3,)  Raw specific force   (for buffer, optional)
        """
        omega_corr = np.asarray(omega_corr, dtype=np.float64)
        accel_corr = np.asarray(accel_corr, dtype=np.float64)
        dt = float(dt)

        # Step 1 — store raw measurements for potential re-integration
        self._raw_buffer.append((
            np.array(omega_raw, dtype=np.float64) if omega_raw is not None else omega_corr.copy(),
            np.array(accel_raw, dtype=np.float64) if accel_raw is not None else accel_corr.copy(),
            dt,
        ))

        # Step 2 — incremental rotation and right Jacobian
        phi_dt = omega_corr * dt           # rotation vector for this step
        dR = exp_so3(phi_dt)               # incremental rotation
        Jr = right_jacobian_so3(phi_dt)    # right Jacobian at phi_dt

        # Step 3 — update bias Jacobians using PRE-STEP delta_R
        # These equations come from Forster TRO 2017, Appendix.
        dR_pre = self.delta_R              # alias for clarity
        dt2 = dt * dt

        self.J_p_ba += self.J_v_ba * dt - 0.5 * dR_pre * dt2
        self.J_p_bg += (
            self.J_v_bg * dt
            - 0.5 * dR_pre @ _skew(accel_corr) @ self.J_R_bg * dt2
        )
        self.J_v_ba += -dR_pre * dt
        self.J_v_bg += -dR_pre @ _skew(accel_corr) @ self.J_R_bg * dt
        self.J_R_bg = dR.T @ self.J_R_bg - Jr * dt

        # Step 4 — update preintegrated state (order: p, v, R)
        self.delta_p += self.delta_v * dt + 0.5 * dR_pre @ accel_corr * dt2
        self.delta_v += dR_pre @ accel_corr * dt
        self.delta_R = dR_pre @ dR

        # Step 5 — propagate covariance using POST-STEP delta_R
        # F is the discrete-time state transition matrix for
        #   [delta_p, delta_v, delta_theta]  (9-dimensional)
        dR_new = self.delta_R

        F = np.eye(9, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * dR_new @ _skew(accel_corr) * dt2
        F[3:6, 6:9] = -dR_new @ _skew(accel_corr) * dt
        F[6:9, 6:9] = dR.T

        # G is the 9×6 noise input matrix for [accel_noise(3), gyro_noise(3)]
        G = np.zeros((9, 6), dtype=np.float64)
        G[0:3, 0:3] = -0.5 * dR_new * dt2
        G[3:6, 0:3] = -dR_new * dt
        G[6:9, 3:6] = -Jr * dt

        # Discrete-time noise covariance
        Q_d = G @ self._Q_c @ G.T

        self.Cov = F @ self.Cov @ F.T + Q_d
        # Enforce symmetry to suppress floating-point drift
        self.Cov = 0.5 * (self.Cov + self.Cov.T)

        # Step 6 — accumulate total time
        self.dt_total += dt

    def bias_corrected_measurement(
        self,
        b_a_new: np.ndarray,
        b_g_new: np.ndarray,
    ):
        """First-order bias correction (Forster TRO 2017, Eq. 44).

        Corrects the stored preintegrated measurements for a small change in
        the linearization-point biases, without full re-integration.

        Parameters
        ----------
        b_a_new : (3,)  Updated accelerometer bias
        b_g_new : (3,)  Updated gyroscope bias

        Returns
        -------
        dp : (3,)  Bias-corrected delta_p
        dv : (3,)  Bias-corrected delta_v
        dR : (3,3) Bias-corrected delta_R
        """
        b_a_new = np.asarray(b_a_new, dtype=np.float64)
        b_g_new = np.asarray(b_g_new, dtype=np.float64)

        db_a = b_a_new - self.b_a
        db_g = b_g_new - self.b_g

        dp = self.delta_p + self.J_p_ba @ db_a + self.J_p_bg @ db_g
        dv = self.delta_v + self.J_v_ba @ db_a + self.J_v_bg @ db_g
        dR = self.delta_R @ exp_so3(self.J_R_bg @ db_g)

        return dp, dv, dR

    def should_reintegrate(
        self,
        b_a_new: np.ndarray,
        b_g_new: np.ndarray,
        thresh_a: float = 0.01,
        thresh_g: float = 0.001,
    ) -> bool:
        """Return True if the bias change exceeds thresholds.

        When the bias change is large, first-order correction is inaccurate
        and full re-integration from the raw buffer is preferable.

        Parameters
        ----------
        b_a_new  : (3,)  New accelerometer bias
        b_g_new  : (3,)  New gyroscope bias
        thresh_a : float Max allowed accel bias change magnitude [m/s^2]
        thresh_g : float Max allowed gyro bias change magnitude  [rad/s]
        """
        b_a_new = np.asarray(b_a_new, dtype=np.float64)
        b_g_new = np.asarray(b_g_new, dtype=np.float64)
        da = np.linalg.norm(b_a_new - self.b_a)
        dg = np.linalg.norm(b_g_new - self.b_g)
        return da > thresh_a or dg > thresh_g

    def reintegrate(self, b_a_new: np.ndarray, b_g_new: np.ndarray) -> None:
        """Full re-integration from the raw measurement buffer with updated biases.

        Resets all accumulated state, then replays every stored raw IMU sample
        with the new linearization-point biases subtracted.

        Parameters
        ----------
        b_a_new : (3,)  New accelerometer bias
        b_g_new : (3,)  New gyroscope bias
        """
        b_a_new = np.asarray(b_a_new, dtype=np.float64)
        b_g_new = np.asarray(b_g_new, dtype=np.float64)

        # Save raw buffer before reset
        raw_buf = list(self._raw_buffer)

        # Reset all accumulated quantities
        self.delta_p = np.zeros(3, dtype=np.float64)
        self.delta_v = np.zeros(3, dtype=np.float64)
        self.delta_R = np.eye(3, dtype=np.float64)
        self.Cov = np.zeros((9, 9), dtype=np.float64)
        self.J_p_ba = np.zeros((3, 3), dtype=np.float64)
        self.J_p_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_v_ba = np.zeros((3, 3), dtype=np.float64)
        self.J_v_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_R_bg = np.zeros((3, 3), dtype=np.float64)
        self.dt_total = 0.0
        self._raw_buffer = []

        # Update linearization point
        self.b_a = b_a_new.copy()
        self.b_g = b_g_new.copy()

        # Replay all stored samples with corrected biases
        for omega_raw, accel_raw, dt in raw_buf:
            omega_corr = omega_raw - self.b_g
            accel_corr = accel_raw - self.b_a
            self.integrate(omega_corr, accel_corr, dt, omega_raw, accel_raw)
