#!/usr/bin/env python3
"""
eskf_node.py
============
Error-State Kalman Filter (ESKF) for Visual-Inertial Odometry.

Fuses bias-corrected IMU measurements (200 Hz) with visual odometry pose
updates (20 Hz) to produce globally-consistent pose and velocity estimates.

Error-state vector  δx ∈ ℝ¹⁵
-------------------------------
  δx = [ δp(3)  δv(3)  δθ(3)  δb_a(3)  δb_g(3) ]
       position velocity  SO(3)  accel-bias  gyro-bias

Subscriptions
-------------
  /imu/processed   sensor_msgs/Imu      bias-corrected + filtered @ 200 Hz
  /vio/odometry    nav_msgs/Odometry    visual 6-DOF pose @ 20 Hz

Publications
------------
  /eskf/odometry   nav_msgs/Odometry    fused pose + velocity (map frame)
  /eskf/pose       geometry_msgs/PoseStamped   convenience pose

Parameters
----------
  config_path    str    path to euroc_params.yaml (IMU noise figures)
  meas_pos_std   float  VIO position noise std-dev  [m]      (default 0.05)
  meas_ang_std   float  VIO rotation noise std-dev  [rad]    (default 0.02)

Theory notes
------------
Prediction  (every IMU sample, dt ≈ 5 ms)
  Nominal state:
    a_w = R·(f − b_a) + g        (world-frame acceleration)
    p  ← p + v·dt + ½·a_w·dt²
    v  ← v + a_w·dt
    q  ← q ⊗ Exp((ω − b_g)·dt)
    b_a, b_g unchanged

  Error-state transition (first-order discrete, F = I + Fc·dt):
    Fc[0:3, 3:6]  =  I             (δṗ = δv)
    Fc[3:6, 6:9]  = -R·[f×]        (δv̇ = -R·[f×]·δθ − R·δb_a)
    Fc[3:6, 9:12] = -R             (δv̇ includes −R·δb_a)
    Fc[6:9, 6:9]  = -[ω×]          (δθ̇ = -[ω×]·δθ − δb_g)
    Fc[6:9,12:15] = -I

  Process noise (discrete, Qd = Qc·dt):
    Qd[3:6,3:6]   = σ_a²·dt·I     (accel white noise → velocity)
    Qd[6:9,6:9]   = σ_g²·dt·I     (gyro  white noise → attitude)
    Qd[9:12,9:12] = σ_ba²·dt·I    (accel bias random walk)
    Qd[12:15,12:15]= σ_bg²·dt·I   (gyro  bias random walk)

  Covariance: P ← F·P·Fᵀ + Qd

Update  (every VIO frame, dt ≈ 50 ms)
  Innovation:
    z = [ p_meas − p_nom ;  Log(R_nom.T @ R_meas) ]   ∈ ℝ⁶

  Measurement Jacobian H (6×15):
    H[0:3, 0:3] = I₃   (position)
    H[3:6, 6:9] = I₃   (orientation — error state is in body frame)

  Joseph-form update (numerically stable):
    S      = H·P·Hᵀ + R_meas
    K      = P·Hᵀ·S⁻¹
    δx     = K·z
    IKH    = I₁₅ − K·H
    P      ← IKH·P·IKHᵀ + K·R_meas·Kᵀ

  Nominal state injection:
    p  ← p + δp
    v  ← v + δv
    q  ← q ⊗ Exp(δθ)   then normalise
    b_a← b_a + δb_a
    b_g← b_g + δb_g
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

import yaml


# ── SO(3) / quaternion utilities ───────────────────────────────────────────────


def _skew(v: np.ndarray) -> np.ndarray:
    """3-vector → 3×3 skew-symmetric (cross-product) matrix."""
    return np.array(
        [[0.0,  -v[2],  v[1]],
         [v[2],  0.0,  -v[0]],
         [-v[1], v[0],  0.0]],
        dtype=np.float64,
    )


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] → 3×3 rotation matrix (body → world)."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2*(y*y + z*z),  2*(x*y - w*z),   2*(x*z + w*y)],
            [2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),      2*(y*z + w*x),   1 - 2*(x*x + y*y)],
        ],
        dtype=np.float64,
    )


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


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, both [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ],
        dtype=np.float64,
    )


def _exp_so3(phi: np.ndarray) -> np.ndarray:
    """
    SO(3) exponential map: rotation vector φ → quaternion [x, y, z, w].

    q = [ sin(‖φ‖/2)·φ̂ ;  cos(‖φ‖/2) ]
    For small ‖φ‖ uses first-order approximation to avoid division by zero.
    """
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        # q ≈ [φ/2, 1] (normalised)
        half = phi * 0.5
        return np.array([half[0], half[1], half[2], 1.0]) / np.sqrt(
            1.0 + 0.25 * angle * angle
        )
    axis = phi / angle
    s = np.sin(angle * 0.5)
    return np.array(
        [axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle * 0.5)],
        dtype=np.float64,
    )


def _log_so3(R: np.ndarray) -> np.ndarray:
    """
    SO(3) logarithm map: rotation matrix → rotation vector φ (3,).

    φ = θ·axis,  cos θ = (trace(R) − 1) / 2
    """
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )


# ── ESKF Node ──────────────────────────────────────────────────────────────────


class EskfNode(Node):
    """
    Error-State Kalman Filter node.

    State machine
    -------------
      UNINIT   Waiting for the first VIO pose to seed the nominal state.
      RUNNING  IMU prediction + VIO update loop active.
    """

    _UNINIT  = "uninit"
    _RUNNING = "running"

    def __init__(self) -> None:
        super().__init__("eskf_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("config_path",  "")
        self.declare_parameter("meas_pos_std", 0.05)   # m
        self.declare_parameter("meas_ang_std", 0.02)   # rad

        config_path  = self.get_parameter("config_path").value
        meas_pos_std = float(self.get_parameter("meas_pos_std").value)
        meas_ang_std = float(self.get_parameter("meas_ang_std").value)

        # ── IMU noise figures (EuRoC defaults, overridden by YAML) ───────────
        self._gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        sigma_a  = 2.0e-3       # accel  noise density  [m/s²/√Hz]
        sigma_g  = 1.6968e-4    # gyro   noise density  [rad/s/√Hz]
        sigma_ba = 3.0e-3       # accel  bias walk      [m/s³/√Hz]
        sigma_bg = 1.9393e-5    # gyro   bias walk      [rad/s²/√Hz]

        if config_path:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                self._gravity = np.array(
                    cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
                )
                imu = cfg.get("imu", {})
                sigma_a  = float(imu.get("accel_noise", sigma_a))
                sigma_g  = float(imu.get("gyro_noise",  sigma_g))
                sigma_ba = float(imu.get("accel_walk",  sigma_ba))
                sigma_bg = float(imu.get("gyro_walk",   sigma_bg))
                self.get_logger().info(f"ESKF config loaded from '{config_path}'.")
            except Exception as exc:
                self.get_logger().warn(
                    f"Could not load config '{config_path}': {exc}. Using defaults."
                )

        # Pre-square noise densities (used every prediction step)
        self._sa2  = sigma_a  ** 2
        self._sg2  = sigma_g  ** 2
        self._sba2 = sigma_ba ** 2
        self._sbg2 = sigma_bg ** 2

        # ── Measurement noise R (6×6) ────────────────────────────────────────
        self._R_meas = np.diag(
            np.concatenate([
                np.full(3, meas_pos_std ** 2),
                np.full(3, meas_ang_std ** 2),
            ])
        )

        # ── Nominal state ────────────────────────────────────────────────────
        self._p   = np.zeros(3, dtype=np.float64)     # position  (world)
        self._v   = np.zeros(3, dtype=np.float64)     # velocity  (world)
        self._q   = np.array([0.0, 0.0, 0.0, 1.0])   # attitude  [x,y,z,w]
        self._b_a = np.zeros(3, dtype=np.float64)     # residual accel bias
        self._b_g = np.zeros(3, dtype=np.float64)     # residual gyro  bias

        # ── Error-state covariance P (15×15) ────────────────────────────────
        # Seeded with generous uncertainty; will contract after first VIO update.
        self._P = np.diag(
            np.concatenate([
                np.full(3, 1.0),     # δp  [m²]
                np.full(3, 1.0),     # δv  [m²/s²]
                np.full(3, 1e-2),    # δθ  [rad²]
                np.full(3, 1e-4),    # δb_a [m²/s⁴]
                np.full(3, 1e-6),    # δb_g [rad²/s²]
            ])
        ).astype(np.float64)

        # ── State machine ────────────────────────────────────────────────────
        self._state         = self._UNINIT
        self._last_stamp_ns: int | None = None
        self._update_count  = 0

        # ── Raw IMU buffer for gravity estimation ─────────────────────────
        # Collect raw /imu0 accel during the static init window so we can
        # compute the true gravity vector in whatever world frame VIO uses.
        # Capped at 2000 samples (10 s @ 200 Hz) — first samples are static.
        self._raw_accel_buf: list[np.ndarray] = []
        self._RAW_BUF_MAX = 2000

        # ── QoS ─────────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self._pub_odom = self.create_publisher(Odometry,    "/eskf/odometry", qos_be)
        self._pub_pose = self.create_publisher(PoseStamped, "/eskf/pose",     qos_be)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Imu,      "/imu0",          self._raw_imu_cb, qos_be)
        self.create_subscription(Imu,      "/imu/processed", self._imu_cb,     qos_be)
        self.create_subscription(Odometry, "/vio/odometry",  self._vio_cb,     10)

        self.get_logger().info(
            "EskfNode ready — waiting for first VIO measurement to initialise."
        )

    # ── Raw IMU gravity buffer ──────────────────────────────────────────────────

    def _raw_imu_cb(self, msg: Imu) -> None:
        """Buffer raw accelerometer data during static init for gravity estimation."""
        if self._state != self._UNINIT or len(self._raw_accel_buf) >= self._RAW_BUF_MAX:
            return
        self._raw_accel_buf.append(np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=np.float64))

    # ── IMU prediction ─────────────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu) -> None:
        if self._state != self._RUNNING:
            return

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        dt = (stamp_ns - self._last_stamp_ns) * 1e-9
        self._last_stamp_ns = stamp_ns

        if dt <= 0.0 or dt > 0.1:
            return

        gyro  = np.array([msg.angular_velocity.x,
                           msg.angular_velocity.y,
                           msg.angular_velocity.z], dtype=np.float64)
        accel = np.array([msg.linear_acceleration.x,
                           msg.linear_acceleration.y,
                           msg.linear_acceleration.z], dtype=np.float64)

        self._predict(gyro, accel, dt)
        self._publish(msg.header)

    # ── VIO update ─────────────────────────────────────────────────────────────

    def _vio_cb(self, msg: Odometry) -> None:
        p_meas = np.array([msg.pose.pose.position.x,
                            msg.pose.pose.position.y,
                            msg.pose.pose.position.z], dtype=np.float64)
        q_meas = np.array([msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w], dtype=np.float64)
        q_meas /= np.linalg.norm(q_meas)

        if self._state == self._UNINIT:
            # Seed nominal state from the first VIO measurement.
            self._p = p_meas.copy()
            self._q = q_meas.copy()
            stamp_ns = (msg.header.stamp.sec * 1_000_000_000
                        + msg.header.stamp.nanosec)
            self._last_stamp_ns = stamp_ns

            # ── Estimate gravity in the VIO world frame ───────────────────
            # The raw /imu0 buffer was collected during the static init window
            # (before the drone started moving).  The mean specific-force in
            # the body frame equals gravity reaction: f_static = -g_body.
            # Rotating to the VIO world frame via the initial VIO quaternion:
            #   g_world = R_wb_init @ (-f_static_body)
            # This avoids the z-up assumption in the config gravity vector.
            if len(self._raw_accel_buf) >= 10:
                R_init = _quat_to_rot(q_meas)           # body → VIO world
                f_static = np.mean(self._raw_accel_buf, axis=0)
                self._gravity = R_init @ (-f_static)    # gravitational accel in world
                self.get_logger().info(
                    f"ESKF: estimated gravity in VIO world frame: "
                    f"{np.round(self._gravity, 4)}  "
                    f"(mag={np.linalg.norm(self._gravity):.3f} m/s², "
                    f"from {len(self._raw_accel_buf)} raw samples)"
                )
            else:
                self.get_logger().warn(
                    f"ESKF: insufficient raw IMU samples ({len(self._raw_accel_buf)}) "
                    "for gravity estimation — using config value."
                )
            self._raw_accel_buf = []  # free memory

            self._state = self._RUNNING
            self.get_logger().info(
                f"ESKF initialised from first VIO pose: "
                f"p={np.round(p_meas, 3)}"
            )
            return

        self._update(p_meas, q_meas)

    # ── Prediction step ────────────────────────────────────────────────────────

    def _predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """
        Propagate nominal state and error-state covariance by one IMU step.

        The /imu/processed measurements have already had the static initial
        biases removed by the ImuProcessingNode.  The ESKF's own b_a / b_g
        states track the residual slow drift on top of that correction.

            f_corr = accel − b_a    (corrected specific force, body frame)
            ω_corr = gyro  − b_g    (corrected angular rate,   body frame)
        """
        R = _quat_to_rot(self._q)

        f_corr = accel - self._b_a
        w_corr = gyro  - self._b_g

        # ── Nominal state propagation ──────────────────────────────────────
        a_world = R @ f_corr + self._gravity         # true acceleration, world
        self._p += self._v * dt + 0.5 * a_world * (dt * dt)
        self._v += a_world * dt
        self._q  = _quat_mul(self._q, _exp_so3(w_corr * dt))
        self._q /= np.linalg.norm(self._q)
        # b_a and b_g are unchanged by prediction (driven only by process noise)

        # ── Error-state transition matrix F (15×15) ────────────────────────
        #
        # F = I + Fc·dt  where Fc contains the continuous-time Jacobians.
        F = np.eye(15, dtype=np.float64)

        F[0:3, 3:6]   =  np.eye(3) * dt              # δṗ = δv
        F[3:6, 6:9]   = -R @ _skew(f_corr) * dt      # δv̇ ← −R·[f×]·δθ
        F[3:6, 9:12]  = -R * dt                       # δv̇ ← −R·δb_a
        F[6:9, 6:9]   =  np.eye(3) - _skew(w_corr) * dt  # δθ̇ = −[ω×]·δθ
        F[6:9, 12:15] = -np.eye(3) * dt               # δθ̇ ← −δb_g

        # ── Discrete process noise Qd (15×15) ─────────────────────────────
        #
        # Qd = Qc · dt  (first-order hold approximation)
        # Only non-zero blocks (position gets no direct noise):
        Q = np.zeros((15, 15), dtype=np.float64)
        Q[3:6,   3:6]   = np.eye(3) * (self._sa2  * dt)  # accel noise → δv
        Q[6:9,   6:9]   = np.eye(3) * (self._sg2  * dt)  # gyro  noise → δθ
        Q[9:12,  9:12]  = np.eye(3) * (self._sba2 * dt)  # accel bias walk
        Q[12:15, 12:15] = np.eye(3) * (self._sbg2 * dt)  # gyro  bias walk

        # ── Covariance propagation ─────────────────────────────────────────
        self._P = F @ self._P @ F.T + Q

    # ── Update step ────────────────────────────────────────────────────────────

    def _update(self, p_meas: np.ndarray, q_meas: np.ndarray) -> None:
        """
        VIO-rate measurement update (Joseph-form EKF update).

        Innovation
        ----------
          z_p = p_meas − p_nom                      (position, 3-DOF)
          z_θ = Log(R_nom.T @ R_meas)               (orientation, 3-DOF, body frame)
          z   = [z_p ; z_θ]   ∈ ℝ⁶

        Measurement Jacobian H (6×15)
        -----------------------------
          H[0:3, 0:3] = I₃   ∂p_meas/∂δp
          H[3:6, 6:9] = I₃   ∂θ_meas/∂δθ  (first-order, error in body frame)
        """
        R_nom  = _quat_to_rot(self._q)
        R_meas = _quat_to_rot(q_meas)

        # Innovation
        z_p = p_meas - self._p
        z_r = _log_so3(R_nom.T @ R_meas)   # rotation error expressed in body frame
        z   = np.concatenate([z_p, z_r])    # (6,)

        # Measurement Jacobian
        H = np.zeros((6, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)

        # Innovation covariance and Kalman gain
        S = H @ self._P @ H.T + self._R_meas    # (6×6)
        K = self._P @ H.T @ np.linalg.solve(S.T, np.eye(6)).T   # (15×6)

        # Error-state estimate
        dx = K @ z    # (15,)

        # Nominal state injection
        self._p   += dx[0:3]
        self._v   += dx[3:6]
        self._q    = _quat_mul(self._q, _exp_so3(dx[6:9]))
        self._q   /= np.linalg.norm(self._q)
        self._b_a += dx[9:12]
        self._b_g += dx[12:15]

        # Covariance update — Joseph form for numerical stability:
        #   P ← (I−KH)·P·(I−KH)ᵀ + K·R·Kᵀ
        IKH = np.eye(15) - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ self._R_meas @ K.T

        # Enforce exact symmetry to suppress floating-point drift
        self._P = (self._P + self._P.T) * 0.5

        self._update_count += 1
        if self._update_count % 20 == 0:   # log every ~1 s
            self.get_logger().debug(
                f"ESKF update #{self._update_count}: "
                f"|z_p|={np.linalg.norm(z_p):.4f} m  "
                f"|z_r|={np.degrees(np.linalg.norm(z_r)):.3f}°  "
                f"b_a={np.round(self._b_a, 5)}  "
                f"b_g={np.round(self._b_g, 7)}"
            )

    # ── Publishers ─────────────────────────────────────────────────────────────

    def _publish(self, header) -> None:
        stamp = header.stamp

        # Odometry message
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id  = "base_link"

        odom.pose.pose.position.x    = self._p[0]
        odom.pose.pose.position.y    = self._p[1]
        odom.pose.pose.position.z    = self._p[2]
        odom.pose.pose.orientation.x = self._q[0]
        odom.pose.pose.orientation.y = self._q[1]
        odom.pose.pose.orientation.z = self._q[2]
        odom.pose.pose.orientation.w = self._q[3]

        odom.twist.twist.linear.x = self._v[0]
        odom.twist.twist.linear.y = self._v[1]
        odom.twist.twist.linear.z = self._v[2]

        self._pub_odom.publish(odom)

        # PoseStamped (convenience for RViz)
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose   = odom.pose.pose
        self._pub_pose.publish(pose)


# ── Entry point ────────────────────────────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = EskfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
