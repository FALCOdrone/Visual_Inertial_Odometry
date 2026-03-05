"""
imu_preintegrator.py
====================
On-manifold IMU preintegration between two VIO keyframes.

Implements Forster et al., "IMU Preintegration on Manifold for Efficient
Visual-Inertial Maximum-a-Posteriori Estimation", TRO 2017.

State: (delta_p, delta_v, delta_R) in R^3 x R^3 x SO(3)
Bias Jacobians: d(delta_p, delta_v, delta_theta)/d(b_a, b_g) maintained
via first-order linearization for efficient bias correction.

This is a standalone Python module with NO ROS dependency.
"""
import numpy as np


# ---- SO(3) helper functions ------------------------------------------------

def _skew(v):
    """
    3-vector -> 3x3 skew-symmetric matrix.

    [v]_x such that [v]_x @ u = v x u  (cross product).
    """
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=np.float64)


def exp_so3(phi):
    """
    Rotation vector -> SO(3) rotation matrix (Rodrigues formula).

    R = I + sin(theta)/theta * [phi]_x + (1 - cos(theta))/theta^2 * [phi]_x^2
    where theta = ||phi||.  For small theta, uses first-order approximation.
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3, dtype=np.float64) + _skew(phi)
    axis = phi / angle
    K = _skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def log_so3(R):
    """
    SO(3) rotation matrix -> rotation vector (matrix logarithm).

    phi = theta * axis, where cos(theta) = (trace(R) - 1) / 2.
    For small theta (near identity), uses first-order approximation.
    """
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    return (angle / (2.0 * np.sin(angle))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64)


def right_jacobian_so3(phi):
    """
    Right Jacobian Jr of SO(3).

    Jr(phi) = I - (1 - cos(theta))/theta^2 * [phi]_x
              + (theta - sin(theta))/theta^3 * [phi]_x^2

    Reference: Chirikjian, "Stochastic Models, Information Theory, and
    Lie Groups", Vol 2.  Used for covariance propagation on SO(3).
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64) - 0.5 * _skew(phi)
    K = _skew(phi / angle)
    return (np.eye(3, dtype=np.float64)
            - ((1.0 - np.cos(angle)) / (angle * angle)) * _skew(phi)
            + ((angle - np.sin(angle)) / (angle ** 3)) * (_skew(phi) @ _skew(phi)))


# ---- Quaternion utilities ---------------------------------------------------

def quat_to_rot(q):
    """
    Quaternion [x, y, z, w] -> 3x3 rotation matrix.

    Uses the standard Hamilton convention where w is the scalar part.
    """
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float64)


def rot_to_quat(R):
    """
    3x3 rotation matrix -> quaternion [x, y, z, w] (Shepperd method).

    Chooses the numerically most stable branch based on the diagonal elements
    to avoid division by near-zero values.  Result is always normalized.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
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
    return q / np.linalg.norm(q)


def quat_mul(q1, q2):
    """
    Hamilton quaternion product q1 * q2,  [x, y, z, w] convention.

    If q1 represents R1 and q2 represents R2, then q1*q2 represents R1 @ R2.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)


# ---- IMU Preintegrator class ------------------------------------------------

class ImuPreintegrator:
    """
    On-manifold IMU preintegration between two keyframes.

    Accumulates IMU measurements between keyframes and computes:
      - Preintegrated position delta_p, velocity delta_v, rotation delta_R
      - 9x9 covariance matrix Cov for (delta_p, delta_v, delta_theta)
      - Bias Jacobians for first-order bias correction without re-integration

    Usage:
        preint = ImuPreintegrator(b_a, b_g, sigma_a, sigma_g, sigma_ba, sigma_bg)
        for omega, accel, dt in imu_samples:
            preint.integrate(omega - b_g, accel - b_a, dt, omega, accel)
        dp, dv, dR = preint.bias_corrected_measurement(b_a_new, b_g_new)

    The integrate() method follows Eq. (32)-(34) of Forster TRO 2017 for
    the preintegrated measurements, and Eq. (63) for the covariance.

    Parameters
    ----------
    b_a : array (3,)
        Accelerometer bias at linearization point [m/s^2].
    b_g : array (3,)
        Gyroscope bias at linearization point [rad/s].
    sigma_a : float
        Accelerometer white noise density [m/s^2/sqrt(Hz)].
    sigma_g : float
        Gyroscope white noise density [rad/s/sqrt(Hz)].
    sigma_ba : float
        Accelerometer bias random walk [m/s^3/sqrt(Hz)].
    sigma_bg : float
        Gyroscope bias random walk [rad/s^2/sqrt(Hz)].
    """

    def __init__(self, b_a, b_g, sigma_a, sigma_g, sigma_ba, sigma_bg):
        self.b_a = np.array(b_a, dtype=np.float64)
        self.b_g = np.array(b_g, dtype=np.float64)

        # Preintegrated measurements (body frame, gravity-free)
        self.delta_p = np.zeros(3, dtype=np.float64)
        self.delta_v = np.zeros(3, dtype=np.float64)
        self.delta_R = np.eye(3, dtype=np.float64)

        # Covariance of [delta_p, delta_v, delta_theta] -- 9x9
        self.Cov = np.zeros((9, 9), dtype=np.float64)

        # Bias Jacobians: how preintegrated measurements change with bias
        # d(delta_p)/d(b_a), d(delta_p)/d(b_g), etc.
        self.J_p_ba = np.zeros((3, 3), dtype=np.float64)
        self.J_p_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_v_ba = np.zeros((3, 3), dtype=np.float64)
        self.J_v_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_R_bg = np.zeros((3, 3), dtype=np.float64)

        self.dt_total = 0.0

        # Continuous-time noise covariance for [accel(3), gyro(3)]
        self._Q_c = np.diag(np.array(
            [sigma_a**2, sigma_a**2, sigma_a**2,
             sigma_g**2, sigma_g**2, sigma_g**2], dtype=np.float64))

        # Store noise parameters for potential re-integration
        self._sigma_a = sigma_a
        self._sigma_g = sigma_g
        self._sigma_ba = sigma_ba
        self._sigma_bg = sigma_bg

        # Raw IMU buffer for re-integration on large bias change
        self._raw_buffer = []  # list of (omega_raw, accel_raw, dt)

    def integrate(self, omega_corr, accel_corr, dt, omega_raw=None, accel_raw=None):
        """
        Integrate one IMU sample (already bias-subtracted at linearization point).

        Implements the discrete preintegration update from Forster TRO 2017:
          delta_p += delta_v * dt + 0.5 * delta_R @ accel_corr * dt^2
          delta_v += delta_R @ accel_corr * dt
          delta_R  = delta_R @ Exp(omega_corr * dt)

        The bias Jacobians and covariance are propagated simultaneously.

        Parameters
        ----------
        omega_corr : array (3,)
            Gyroscope measurement minus bias: omega_meas - b_g [rad/s].
        accel_corr : array (3,)
            Accelerometer measurement minus bias: accel_meas - b_a [m/s^2].
        dt : float
            Time step [s].
        omega_raw : array (3,), optional
            Raw gyroscope measurement (stored for re-integration).
        accel_raw : array (3,), optional
            Raw accelerometer measurement (stored for re-integration).
        """
        if omega_raw is not None and accel_raw is not None:
            self._raw_buffer.append((omega_raw.copy(), accel_raw.copy(), dt))

        # Incremental rotation from this gyro sample
        dR = exp_so3(omega_corr * dt)
        Jr = right_jacobian_so3(omega_corr * dt)

        # ---- Update bias Jacobians (BEFORE updating preintegrated state) ----
        # Forster TRO 2017, Eq. (A.5)-(A.9)
        # Note: delta_R here is the value BEFORE this integration step
        self.J_p_ba += self.J_v_ba * dt - 0.5 * self.delta_R * dt**2
        self.J_p_bg += (self.J_v_bg * dt
                        - 0.5 * self.delta_R @ _skew(accel_corr) @ self.J_R_bg * dt**2)
        self.J_v_ba += -self.delta_R * dt
        self.J_v_bg += -self.delta_R @ _skew(accel_corr) @ self.J_R_bg * dt
        self.J_R_bg = dR.T @ self.J_R_bg - Jr * dt

        # ---- Update preintegrated measurements ----
        # Order: p before v before R (p uses old v and R, v uses old R)
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ accel_corr * dt**2
        self.delta_v += self.delta_R @ accel_corr * dt
        self.delta_R = self.delta_R @ dR

        # ---- Propagate covariance ----
        # State-transition matrix F (9x9) for [delta_p, delta_v, delta_theta]
        # Forster TRO 2017, Eq. (63)
        F = np.eye(9, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * self.delta_R @ _skew(accel_corr) * dt**2
        F[3:6, 6:9] = -self.delta_R @ _skew(accel_corr) * dt
        F[6:9, 6:9] = dR.T

        # Noise input matrix G (9x6) mapping [accel_noise, gyro_noise] to state
        G = np.zeros((9, 6), dtype=np.float64)
        G[0:3, 0:3] = -0.5 * self.delta_R * dt**2
        G[3:6, 0:3] = -self.delta_R * dt
        G[6:9, 3:6] = -Jr * dt

        # Discrete noise covariance: Q_d = G @ Q_c @ G^T
        # Q_c contains spectral densities sigma^2; multiplied by dt inside G
        Q_d = G @ self._Q_c @ G.T
        self.Cov = F @ self.Cov @ F.T + Q_d

        self.dt_total += dt

    def bias_corrected_measurement(self, b_a_new, b_g_new):
        """
        First-order bias correction of preintegrated measurements.

        When biases change slightly from the linearization point, we can
        correct the preintegrated quantities without re-integration:
          delta_p_corr = delta_p + J_p_ba @ db_a + J_p_bg @ db_g
          delta_v_corr = delta_v + J_v_ba @ db_a + J_v_bg @ db_g
          delta_R_corr = delta_R @ Exp(J_R_bg @ db_g)

        Parameters
        ----------
        b_a_new : array (3,)
            Updated accelerometer bias.
        b_g_new : array (3,)
            Updated gyroscope bias.

        Returns
        -------
        dp : array (3,)
            Bias-corrected preintegrated position.
        dv : array (3,)
            Bias-corrected preintegrated velocity.
        dR : array (3, 3)
            Bias-corrected preintegrated rotation.
        """
        db_a = b_a_new - self.b_a
        db_g = b_g_new - self.b_g
        dp = self.delta_p + self.J_p_ba @ db_a + self.J_p_bg @ db_g
        dv = self.delta_v + self.J_v_ba @ db_a + self.J_v_bg @ db_g
        dR = self.delta_R @ exp_so3(self.J_R_bg @ db_g)
        return dp, dv, dR

    def should_reintegrate(self, b_a_new, b_g_new, thresh_a=0.01, thresh_g=0.001):
        """
        Check if bias change is large enough to warrant full re-integration.

        Returns True if the accelerometer bias changed by more than thresh_a
        or gyroscope bias changed by more than thresh_g (in norm).
        """
        return (np.linalg.norm(b_a_new - self.b_a) > thresh_a or
                np.linalg.norm(b_g_new - self.b_g) > thresh_g)

    def reintegrate(self, b_a_new, b_g_new):
        """
        Full re-integration from raw IMU buffer with new bias linearization.

        This is more accurate than first-order correction when biases have
        changed significantly.  Uses the stored raw measurements.

        Parameters
        ----------
        b_a_new : array (3,)
            New accelerometer bias linearization point.
        b_g_new : array (3,)
            New gyroscope bias linearization point.
        """
        raw = self._raw_buffer[:]
        sigma_a = self._sigma_a
        sigma_g = self._sigma_g
        sigma_ba = self._sigma_ba
        sigma_bg = self._sigma_bg
        self.__init__(b_a_new, b_g_new, sigma_a, sigma_g, sigma_ba, sigma_bg)
        for omega_raw, accel_raw, dt in raw:
            self.integrate(
                omega_raw - b_g_new, accel_raw - b_a_new, dt,
                omega_raw, accel_raw)
