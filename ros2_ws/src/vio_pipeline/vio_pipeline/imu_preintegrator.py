"""
imu_preintegrator.py
====================
On-manifold IMU preintegration (Forster TRO 2017, discrete-time).

Accumulates IMU measurements between two keyframe times and produces
preintegrated rotation, velocity, and position deltas along with their
covariance and bias Jacobians for first-order bias correction.

Not a ROS node — pure computation, unit-testable standalone.
"""

import numpy as np
from vio_pipeline.so3_utils import skew, exp_so3, right_jacobian_so3


class ImuPreintegrator:
    """Preintegrates IMU measurements between two keyframes."""

    def __init__(
        self,
        b_a: np.ndarray,
        b_g: np.ndarray,
        sigma_a: float = 2.0e-3,
        sigma_g: float = 1.6968e-4,
        sigma_ba: float = 3.0e-3,
        sigma_bg: float = 1.9393e-5,
        gravity: np.ndarray | None = None,
    ):
        self._sigma_a = sigma_a
        self._sigma_g = sigma_g
        self._sigma_ba = sigma_ba
        self._sigma_bg = sigma_bg
        self._gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])

        self._b_a_lin = b_a.copy()
        self._b_g_lin = b_g.copy()
        self.reset(b_a, b_g)

    def reset(self, b_a: np.ndarray, b_g: np.ndarray) -> None:
        """Start a new preintegration interval with given bias linearization point."""
        self._b_a_lin = b_a.copy()
        self._b_g_lin = b_g.copy()

        self._delta_R = np.eye(3, dtype=np.float64)
        self._delta_v = np.zeros(3, dtype=np.float64)
        self._delta_p = np.zeros(3, dtype=np.float64)
        self._delta_t = 0.0

        # 9x9 covariance: [dR(3), dv(3), dp(3)]
        self._covariance = np.zeros((9, 9), dtype=np.float64)

        # Bias Jacobians for first-order correction
        self._d_R_d_bg = np.zeros((3, 3), dtype=np.float64)
        self._d_v_d_ba = np.zeros((3, 3), dtype=np.float64)
        self._d_v_d_bg = np.zeros((3, 3), dtype=np.float64)
        self._d_p_d_ba = np.zeros((3, 3), dtype=np.float64)
        self._d_p_d_bg = np.zeros((3, 3), dtype=np.float64)

    def integrate(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """Integrate a single IMU measurement."""
        if dt <= 0.0:
            return

        w_corr = gyro - self._b_g_lin
        f_corr = accel - self._b_a_lin

        # Rotation increment
        dR_inc = exp_so3(w_corr * dt)
        Jr = right_jacobian_so3(w_corr * dt)

        # --- Covariance propagation ---
        # State: [dtheta, dv, dp]
        # Continuous-time Jacobian A and noise input B
        A = np.zeros((9, 9), dtype=np.float64)
        A[0:3, 0:3] = -skew(w_corr)
        A[3:6, 0:3] = -self._delta_R @ skew(f_corr)
        A[6:9, 3:6] = np.eye(3)

        B = np.zeros((9, 6), dtype=np.float64)
        B[0:3, 0:3] = -Jr  # gyro noise -> rotation
        B[3:6, 3:6] = -self._delta_R  # accel noise -> velocity

        # Discrete transition
        F_d = np.eye(9, dtype=np.float64) + A * dt
        Q_d = np.diag([
            self._sigma_g**2, self._sigma_g**2, self._sigma_g**2,
            self._sigma_a**2, self._sigma_a**2, self._sigma_a**2,
        ]) * dt

        self._covariance = F_d @ self._covariance @ F_d.T + B @ Q_d @ B.T
        # Enforce symmetry
        self._covariance = (self._covariance + self._covariance.T) * 0.5

        # --- Bias Jacobians ---
        # d_R_d_bg: propagation
        self._d_R_d_bg = dR_inc.T @ self._d_R_d_bg - Jr * dt

        # d_v_d_ba, d_v_d_bg
        self._d_v_d_ba = self._d_v_d_ba + (-self._delta_R) * dt  # -dR @ I * dt
        self._d_v_d_bg = self._d_v_d_bg + (-self._delta_R @ skew(f_corr) @ self._d_R_d_bg) * dt

        # d_p_d_ba, d_p_d_bg
        self._d_p_d_ba = self._d_p_d_ba + self._d_v_d_ba * dt + 0.5 * (-self._delta_R) * dt * dt
        self._d_p_d_bg = self._d_p_d_bg + self._d_v_d_bg * dt + 0.5 * (-self._delta_R @ skew(f_corr) @ self._d_R_d_bg) * dt * dt

        # --- State propagation ---
        # Order matters: use current delta_R before updating it
        self._delta_p = self._delta_p + self._delta_v * dt + 0.5 * self._delta_R @ f_corr * dt * dt
        self._delta_v = self._delta_v + self._delta_R @ f_corr * dt
        self._delta_R = self._delta_R @ dR_inc

        self._delta_t += dt

    def correct(self, new_b_a: np.ndarray, new_b_g: np.ndarray):
        """First-order bias correction (read-only, returns corrected deltas)."""
        db_a = new_b_a - self._b_a_lin
        db_g = new_b_g - self._b_g_lin

        # Corrected rotation
        dR_corr = self._delta_R @ exp_so3(self._d_R_d_bg @ db_g)

        # Corrected velocity and position
        dv_corr = self._delta_v + self._d_v_d_ba @ db_a + self._d_v_d_bg @ db_g
        dp_corr = self._delta_p + self._d_p_d_ba @ db_a + self._d_p_d_bg @ db_g

        return dR_corr, dv_corr, dp_corr

    # --- Properties ---
    @property
    def delta_R(self) -> np.ndarray:
        return self._delta_R

    @property
    def delta_v(self) -> np.ndarray:
        return self._delta_v

    @property
    def delta_p(self) -> np.ndarray:
        return self._delta_p

    @property
    def delta_t(self) -> float:
        return self._delta_t

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def d_R_d_bg(self) -> np.ndarray:
        return self._d_R_d_bg

    @property
    def d_v_d_ba(self) -> np.ndarray:
        return self._d_v_d_ba

    @property
    def d_v_d_bg(self) -> np.ndarray:
        return self._d_v_d_bg

    @property
    def d_p_d_ba(self) -> np.ndarray:
        return self._d_p_d_ba

    @property
    def d_p_d_bg(self) -> np.ndarray:
        return self._d_p_d_bg

    @property
    def b_a_lin(self) -> np.ndarray:
        return self._b_a_lin

    @property
    def b_g_lin(self) -> np.ndarray:
        return self._b_g_lin

    @property
    def gravity(self) -> np.ndarray:
        return self._gravity
