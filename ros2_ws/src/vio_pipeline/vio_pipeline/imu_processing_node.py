#!/usr/bin/env python3
"""
imu_processing_node.py
======================
IMU preprocessing pipeline for VIO.

Subscribes to raw IMU measurements (sensor_msgs/Imu) and performs:
  1. Static initialisation — collects `init_duration` seconds of samples to
     estimate gyroscope bias (mean ω at rest) and accelerometer bias
     (deviation from expected gravity reaction), and to derive initial
     roll/pitch from the gravity vector.
  2. Bias subtraction from every incoming measurement.
  3. 2nd-order Butterworth low-pass filtering (gyro and accel independently).
  4. Dead-reckoning integration: quaternion attitude, velocity, position.

Published topics
----------------
  /imu/processed   sensor_msgs/Imu      bias-corrected + filtered measurements
  /imu/odometry    nav_msgs/Odometry    dead-reckoning pose (drifts without fusion)

Parameters
----------
  config_path       str    path to euroc_params.yaml  (default: "")
  imu_topic         str    raw IMU input topic         (default: "/imu0")
  init_duration     float  seconds of static data for bias init  (default: 2.0)
  gyro_lpf_cutoff   float  gyro LPF cutoff in Hz       (default: 50.0)
  accel_lpf_cutoff  float  accel LPF cutoff in Hz      (default: 30.0)
  imu_rate_hz       int    expected IMU sample rate     (default: 200)
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

import yaml


# ── 2nd-order Butterworth biquad (no scipy) ───────────────────────────────────
#
# Coefficients derived analytically via the bilinear transform with
# frequency pre-warping.  For cutoff fc (Hz) at sample rate fs (Hz):
#
#   K    = tan(π · fc / fs)          ← pre-warped analog cutoff
#   norm = 1 + √2·K + K²
#   b0   = K² / norm,  b1 = 2·b0,  b2 = b0
#   a1   = 2·(K²−1) / norm
#   a2   = (1 − √2·K + K²) / norm
#
# This gives DC gain = 1 and −3 dB exactly at fc.


def _butter2_lp(fc: float, fs: float) -> tuple:
    """Return (b0, b1, b2, a1, a2) for a 2nd-order Butterworth LP biquad."""
    K = np.tan(np.pi * fc / fs)
    K2 = K * K
    norm = 1.0 + np.sqrt(2.0) * K + K2
    b0 = K2 / norm
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (K2 - 1.0) / norm
    a2 = (1.0 - np.sqrt(2.0) * K + K2) / norm
    return b0, b1, b2, a1, a2


class _Biquad:
    """
    Single-channel Direct Form II biquad IIR filter.

    Update equations (per sample):
        w[n] = x[n] − a1·w[n−1] − a2·w[n−2]
        y[n] = b0·w[n] + b1·w[n−1] + b2·w[n−2]
    """

    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float) -> None:
        self.b0, self.b1, self.b2 = b0, b1, b2
        self.a1, self.a2 = a1, a2
        self._w1 = self._w2 = 0.0  # delay states w[n-1], w[n-2]

    def seed(self, x0: float) -> None:
        """
        Initialise delay state for a constant input x0 so the first output
        equals x0 with no startup transient.

        At steady state: w_ss = x0 / (1 + a1 + a2)
        """
        denom = 1.0 + self.a1 + self.a2
        w_ss = x0 / denom if abs(denom) > 1e-15 else 0.0
        self._w1 = self._w2 = w_ss

    def step(self, x: float) -> float:
        """Process one sample and return the filtered output."""
        w = x - self.a1 * self._w1 - self.a2 * self._w2
        y = self.b0 * w + self.b1 * self._w1 + self.b2 * self._w2
        self._w2 = self._w1
        self._w1 = w
        return y


class _Biquad3:
    """Three independent biquad channels sharing the same coefficients (x/y/z)."""

    def __init__(self, fc: float, fs: float) -> None:
        coeffs = _butter2_lp(fc, fs)
        self._ch = [_Biquad(*coeffs) for _ in range(3)]

    def seed(self, vec: np.ndarray) -> None:
        for i, ch in enumerate(self._ch):
            ch.seed(float(vec[i]))

    def step(self, vec: np.ndarray) -> np.ndarray:
        return np.array([ch.step(float(vec[i])) for i, ch in enumerate(self._ch)])


# ── Quaternion / rotation helpers ─────────────────────────────────────────────


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [x, y, z, w] (Shepperd method)."""
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
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] → 3×3 rotation matrix (body → world)."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2.  Both quaternions are [x, y, z, w]."""
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


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit rotation axis + angle (rad) → quaternion [x, y, z, w]."""
    s = np.sin(angle / 2.0)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)], dtype=np.float64
    )


def _gravity_to_quat(g_meas_body: np.ndarray) -> np.ndarray:
    """
    Estimate initial world-from-body quaternion (roll + pitch only; yaw = 0)
    from the gravity reaction measured in the body frame.

    Physics recap
    -------------
    At static rest the accelerometer measures the reaction force of the
    surface (equal in magnitude to gravity, pointing *up* in world frame):

        a_body = R_{bw} · [0, 0, +|g|]^T

    We find the rotation q_{wb} (body → world) that maps g_hat_body to
    world +Z = [0, 0, 1].  This sets roll and pitch; yaw is left at zero
    (unobservable from accelerometer alone).
    """
    g_hat = g_meas_body / np.linalg.norm(g_meas_body)
    z_world = np.array([0.0, 0.0, 1.0])

    cos_a = float(np.clip(np.dot(g_hat, z_world), -1.0, 1.0))
    axis = np.cross(g_hat, z_world)
    sin_a = np.linalg.norm(axis)

    if sin_a < 1e-8:
        # Vectors already aligned or anti-aligned
        if cos_a > 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0])  # identity
        else:
            return np.array([1.0, 0.0, 0.0, 0.0])  # 180° around x

    return _axis_angle_to_quat(axis / sin_a, np.arctan2(sin_a, cos_a))


# ── Node ──────────────────────────────────────────────────────────────────────


class ImuProcessingNode(Node):
    """
    IMU preprocessing pipeline.

    State machine
    -------------
      INIT    Collect init_duration seconds of static data.
              On completion: estimate biases, set initial orientation.
      RUNNING Bias-subtract → LPF → integrate → publish.
    """

    _INIT = "init"
    _RUNNING = "running"

    def __init__(self):
        super().__init__("imu_processing_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("config_path", "")
        self.declare_parameter("imu_topic", "/imu0")
        self.declare_parameter("init_duration", 2.0)
        self.declare_parameter("gyro_lpf_cutoff", 50.0)
        self.declare_parameter("accel_lpf_cutoff", 30.0)
        self.declare_parameter("imu_rate_hz", 200)
        config_path = self.get_parameter("config_path").value
        imu_topic = self.get_parameter("imu_topic").value
        self._init_dur = self.get_parameter("init_duration").value
        gyro_cut = self.get_parameter("gyro_lpf_cutoff").value
        accel_cut = self.get_parameter("accel_lpf_cutoff").value
        imu_rate_hz = self.get_parameter("imu_rate_hz").value

        # ── Load YAML config (optional) ──────────────────────────────────────
        self._gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        if config_path:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                self._gravity = np.array(
                    cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
                )
                imu_rate_hz = int(cfg.get("imu", {}).get("rate_hz", imu_rate_hz))
                self.get_logger().info(f"Config loaded from '{config_path}'.")
            except Exception as exc:
                self.get_logger().warn(
                    f"Could not load config '{config_path}': {exc}. Using defaults."
                )

        self._imu_rate_hz = imu_rate_hz

        # ── Butterworth low-pass filter (2nd order, pure-numpy biquad) ───────
        #
        # Coefficients computed analytically via bilinear transform.
        # Delay states are seeded from the first bias-corrected sample so there
        # is no startup transient.
        #
        # Cutoffs: gyro 50 Hz, accel 30 Hz  (at 200 Hz IMU rate).
        self._filt_gyro = _Biquad3(gyro_cut, imu_rate_hz)
        self._filt_accel = _Biquad3(accel_cut, imu_rate_hz)

        # ── State machine ────────────────────────────────────────────────────
        self._state = self._INIT
        self._init_gyro_buf: list[np.ndarray] = []
        self._init_accel_buf: list[np.ndarray] = []
        self._init_samples_needed = max(1, int(self._init_dur * imu_rate_hz))

        # Bias estimates (zero until init completes)
        self._bias_gyro = np.zeros(3)
        self._bias_accel = np.zeros(3)

        # Dead-reckoning state
        self._q: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])  # [x,y,z,w]
        self._vel: np.ndarray = np.zeros(3)
        self._pos: np.ndarray = np.zeros(3)
        self._last_stamp_ns: int | None = None

        # ── QoS ─────────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self._pub_imu = self.create_publisher(Imu, "/imu/processed", qos)
        self._pub_odom = self.create_publisher(Odometry, "/imu/odometry", qos)

        # ── Subscriber ───────────────────────────────────────────────────────
        self.create_subscription(Imu, imu_topic, self._imu_callback, qos)

        self.get_logger().info(
            f"ImuProcessingNode ready — subscribing to '{imu_topic}'. "
            f"Collecting {self._init_dur:.1f} s of static data "
            f"({self._init_samples_needed} samples) for bias initialisation…"
        )

    # ── IMU callback ──────────────────────────────────────────────────────────

    def _imu_callback(self, msg: Imu) -> None:
        gyro_raw = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=np.float64,
        )
        accel_raw = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=np.float64,
        )
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # ── INIT phase ───────────────────────────────────────────────────────
        if self._state == self._INIT:
            self._init_gyro_buf.append(gyro_raw.copy())
            self._init_accel_buf.append(accel_raw.copy())
            if len(self._init_gyro_buf) >= self._init_samples_needed:
                self._finalize_init(stamp_ns)
            return

        # ── RUNNING phase ─────────────────────────────────────────────────────
        dt = (stamp_ns - self._last_stamp_ns) * 1e-9
        self._last_stamp_ns = stamp_ns

        if dt <= 0.0 or dt > 0.1:
            # Drop stale, out-of-order, or very delayed packets
            return

        # 1. Bias subtraction
        gyro_corr = gyro_raw - self._bias_gyro
        accel_corr = accel_raw - self._bias_accel

        # 2. Low-pass filter (single-sample, state-preserving)
        gyro_filt = self._filt_gyro.step(gyro_corr)
        accel_filt = self._filt_accel.step(accel_corr)

        # 3. Dead-reckoning integration
        self._integrate(gyro_filt, accel_filt, dt)

        # 4. Publish
        self._publish_imu(msg.header, gyro_filt, accel_filt)
        self._publish_odometry(msg.header)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _finalize_init(self, stamp_ns: int) -> None:
        """
        Compute biases from the static window and set initial orientation.

        Gyroscope bias
        --------------
          b_g = mean(ω_raw)
          At rest the body does not rotate → ideal ω = 0.

        Accelerometer bias
        ------------------
          At rest:  a_raw = R_{bw} · [0, 0, +|g|]^T + b_a
          Solving:  b_a = mean(a_raw) − R_{bw} · [0, 0, +|g|]^T
          where R_{bw} = R_{wb}^T is determined from the initial orientation.

        Initial orientation
        -------------------
          Roll and pitch from the gravity direction in the body frame.
          Yaw is set to zero (unobservable without a magnetometer).
        """
        n = len(self._init_gyro_buf)
        gyro_arr = np.array(self._init_gyro_buf, dtype=np.float64)  # (N, 3)
        accel_arr = np.array(self._init_accel_buf, dtype=np.float64)

        # Gyro bias
        self._bias_gyro = gyro_arr.mean(axis=0)

        # Gravity sanity check
        g_mean = accel_arr.mean(axis=0)
        g_mag = np.linalg.norm(g_mean)
        g_expected = np.linalg.norm(self._gravity)
        if abs(g_mag - g_expected) / g_expected > 0.10:
            self.get_logger().warn(
                f"Measured gravity magnitude {g_mag:.3f} m/s² differs >10 % from "
                f"expected {g_expected:.3f} m/s². "
                "Verify the platform is stationary during initialisation."
            )

        # Initial orientation: roll + pitch from gravity; yaw = 0
        self._q = _gravity_to_quat(g_mean)

        # Accelerometer bias
        R_wb = _quat_to_rot(self._q)  # body → world
        R_bw = R_wb.T  # world → body
        g_reaction_world = -self._gravity  # [0, 0, +9.81]  (reaction force)
        self._bias_accel = g_mean - R_bw @ g_reaction_world

        # Seed filter delays from the mean of the static window, not the last
        # sample (which may be a motor-vibration peak if motors started just
        # before init ended, causing a false startup transient).
        g0 = gyro_arr.mean(axis=0) - self._bias_gyro
        a0 = accel_arr.mean(axis=0) - self._bias_accel
        self._filt_gyro.seed(g0)
        self._filt_accel.seed(a0)

        self._last_stamp_ns = stamp_ns
        self._state = self._RUNNING

        roll_deg, pitch_deg = self._roll_pitch_deg()
        self.get_logger().info(
            f"Bias initialisation complete ({n} samples, "
            f"{n / self._imu_rate_hz:.1f} s).\n"
            f"  gyro  bias : {np.round(self._bias_gyro,  6)} rad/s\n"
            f"  accel bias : {np.round(self._bias_accel, 4)} m/s²\n"
            f"  gravity mag: {g_mag:.4f} m/s²\n"
            f"  init roll  : {roll_deg:.2f}°\n"
            f"  init pitch : {pitch_deg:.2f}°"
        )

        # Free buffer memory
        self._init_gyro_buf = []
        self._init_accel_buf = []

    # ── Dead-reckoning integration ────────────────────────────────────────────

    def _integrate(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """
        Propagate orientation, velocity, and position by one IMU step.

        Orientation (SO(3) exponential map)
        ------------------------------------
          δθ = ω · dt   (incremental rotation vector in body frame)
          q  ← q ⊗ Exp(δθ)   then re-normalise.

        Linear kinematics (Euler, mid-point gravity correction)
        --------------------------------------------------------
          The accelerometer measures *specific force* (reaction force minus
          gravity). To obtain true acceleration in the world frame:

              a_world = R_{wb} · f_body + g_world

          where g_world = [0, 0, −9.81] so adding it cancels the gravity
          component that the sensor always sees.

          v ← v + a_world · dt
          p ← p + v · dt + ½ · a_world · dt²

        Note: This integration will drift. It is intended as a preview for
        the upcoming ESKF, which will correct the drift with visual updates.
        """
        # Quaternion integration
        dtheta = gyro * dt
        angle = np.linalg.norm(dtheta)
        if angle > 1e-10:
            dq = _axis_angle_to_quat(dtheta / angle, angle)
            self._q = _quat_mul(self._q, dq)
            self._q /= np.linalg.norm(self._q)  # numerical normalisation

        # True world-frame acceleration (gravity removal)
        R_wb = _quat_to_rot(self._q)
        a_world = R_wb @ accel + self._gravity  # gravity = [0, 0, -9.81]

        self._pos += self._vel * dt + 0.5 * a_world * (dt * dt)
        self._vel += a_world * dt

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_imu(self, header, gyro: np.ndarray, accel: np.ndarray) -> None:
        out = Imu()
        out.header = header
        out.header.frame_id = "imu"

        out.angular_velocity.x = gyro[0]
        out.angular_velocity.y = gyro[1]
        out.angular_velocity.z = gyro[2]

        out.linear_acceleration.x = accel[0]
        out.linear_acceleration.y = accel[1]
        out.linear_acceleration.z = accel[2]

        # Orientation not estimated in this node
        out.orientation_covariance[0] = -1.0
        out.angular_velocity_covariance[0] = -1.0
        out.linear_acceleration_covariance[0] = -1.0

        self._pub_imu.publish(out)

    def _publish_odometry(self, header) -> None:
        out = Odometry()
        out.header = header
        out.header.frame_id = "map"
        out.child_frame_id = "imu"

        out.pose.pose.position.x = self._pos[0]
        out.pose.pose.position.y = self._pos[1]
        out.pose.pose.position.z = self._pos[2]
        out.pose.pose.orientation.x = self._q[0]
        out.pose.pose.orientation.y = self._q[1]
        out.pose.pose.orientation.z = self._q[2]
        out.pose.pose.orientation.w = self._q[3]

        out.twist.twist.linear.x = self._vel[0]
        out.twist.twist.linear.y = self._vel[1]
        out.twist.twist.linear.z = self._vel[2]

        self._pub_odom.publish(out)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _roll_pitch_deg(self) -> tuple[float, float]:
        R = _quat_to_rot(self._q)
        pitch = np.degrees(np.arcsin(-R[2, 0]))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        return roll, pitch


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = ImuProcessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
