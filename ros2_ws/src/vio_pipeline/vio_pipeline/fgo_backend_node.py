"""
fgo_backend_node.py
===================
ROS2 node: Factor Graph Optimization (FGO) backend for VIO.

Subscribes to IMU and VIO odometry, performs sliding-window optimization
with IMU preintegration and VO relative-pose factors, and publishes
the optimized pose.

Replaces the ESKF when launched with use_fgo:=true.
"""

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

from vio_pipeline.so3_utils import (
    skew, quat_to_rot, rot_to_quat, quat_mul, exp_so3, exp_so3_quat, log_so3,
)
from vio_pipeline.imu_preintegrator import ImuPreintegrator
from vio_pipeline.factor_graph import SlidingWindowGraph, KeyframeState


class FgoBackendNode(Node):
    """Sliding-window FGO backend node."""

    _UNINIT = "uninit"
    _RUNNING = "running"

    def __init__(self) -> None:
        super().__init__("fgo_backend_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("config_path", "")
        self.declare_parameter("window_size", 10)
        self.declare_parameter("lm_max_iter", 5)
        self.declare_parameter("lm_lambda_init", 1e-3)
        self.declare_parameter("vo_pos_std", 0.05)
        self.declare_parameter("vo_ang_std", 0.05)
        self.declare_parameter("prior_pos_std", 0.01)
        self.declare_parameter("prior_vel_std", 1.0)
        self.declare_parameter("prior_att_std", 0.01)
        self.declare_parameter("prior_ba_std", 0.02)
        self.declare_parameter("prior_bg_std", 5.0e-4)
        self.declare_parameter("imu_noise_scale", 20.0)
        self.declare_parameter("min_gravity_samples", 100)

        config_path = self.get_parameter("config_path").value
        self._window_size = int(self.get_parameter("window_size").value)
        self._lm_max_iter = int(self.get_parameter("lm_max_iter").value)
        self._lm_lambda_init = float(self.get_parameter("lm_lambda_init").value)
        self._vo_pos_std = float(self.get_parameter("vo_pos_std").value)
        self._vo_ang_std = float(self.get_parameter("vo_ang_std").value)
        self._prior_pos_std = float(self.get_parameter("prior_pos_std").value)
        self._prior_vel_std = float(self.get_parameter("prior_vel_std").value)
        self._prior_att_std = float(self.get_parameter("prior_att_std").value)
        self._prior_ba_std = float(self.get_parameter("prior_ba_std").value)
        self._prior_bg_std = float(self.get_parameter("prior_bg_std").value)
        self._imu_noise_scale = float(self.get_parameter("imu_noise_scale").value)
        self._min_gravity_samples = int(self.get_parameter("min_gravity_samples").value)

        # ── IMU noise figures ─────────────────────────────────────────────────
        self._gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self._sigma_a_raw = 2.0e-3
        self._sigma_g_raw = 1.6968e-4
        self._sigma_ba = 3.0e-3
        self._sigma_bg = 1.9393e-5

        if config_path:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                self._gravity = np.array(
                    cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
                )
                imu = cfg.get("imu", {})
                self._sigma_a_raw = float(imu.get("accel_noise", self._sigma_a_raw))
                self._sigma_g_raw = float(imu.get("gyro_noise", self._sigma_g_raw))
                self._sigma_ba = float(imu.get("accel_walk", self._sigma_ba))
                self._sigma_bg = float(imu.get("gyro_walk", self._sigma_bg))
                self.get_logger().info(f"FGO config loaded from '{config_path}'.")
            except Exception as exc:
                self.get_logger().warn(
                    f"Could not load config '{config_path}': {exc}. Using defaults."
                )

        # Scale IMU white noise for preintegration (standard practice for
        # optimization-based VIO — datasheet values are too tight, causing
        # IMU factors to dominate over VO factors; VINS-Mono uses ~20-40x)
        self._sigma_a = self._sigma_a_raw * self._imu_noise_scale
        self._sigma_g = self._sigma_g_raw * self._imu_noise_scale

        self.get_logger().info(
            f"FGO params: window={self._window_size} lm_iter={self._lm_max_iter} "
            f"lm_lambda={self._lm_lambda_init} "
            f"vo_pos_std={self._vo_pos_std} vo_ang_std={self._vo_ang_std} "
            f"imu_noise_scale={self._imu_noise_scale} "
            f"sigma_a={self._sigma_a:.4f} sigma_g={self._sigma_g:.5f}"
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._state = self._UNINIT
        self._graph: SlidingWindowGraph | None = None
        self._preint: ImuPreintegrator | None = None

        # Previous VIO message for relative pose computation
        self._prev_vio_p: np.ndarray | None = None
        self._prev_vio_R: np.ndarray | None = None
        self._prev_vio_stamp_ns: int | None = None

        # IMU buffer: list of (stamp_ns, gyro(3,), accel(3,))
        self._imu_buf: list[tuple[int, np.ndarray, np.ndarray]] = []

        # Raw IMU buffer for gravity estimation (during UNINIT)
        self._raw_accel_buf: list[np.ndarray] = []
        self._RAW_BUF_MAX = 2000

        # First VIO message (saved for gravity computation with correct R)
        self._first_vio: dict | None = None

        # Keyframe counter for logging
        self._kf_count = 0

        # ── QoS ───────────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_odom = self.create_publisher(Odometry, "/fgo/odometry", qos_be)
        self._pub_pose = self.create_publisher(PoseStamped, "/fgo/pose", qos_be)
        self._pub_path = self.create_publisher(Path, "/fgo/path", qos_be)
        self._path_msg = Path()
        self._path_msg.header.frame_id = "map"

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Imu, "/imu0", self._raw_imu_cb, qos_be)
        self.create_subscription(Imu, "/imu/processed", self._imu_cb, qos_be)
        self.create_subscription(Odometry, "/vio/odometry", self._vio_cb, 10)

        self.get_logger().info("FgoBackendNode ready — waiting for first VIO keyframe.")

    # ── Raw IMU for gravity estimation ────────────────────────────────────────

    def _raw_imu_cb(self, msg: Imu) -> None:
        if self._state != self._UNINIT or len(self._raw_accel_buf) >= self._RAW_BUF_MAX:
            return
        self._raw_accel_buf.append(np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=np.float64))

    # ── IMU buffer ────────────────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=np.float64)
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=np.float64)
        self._imu_buf.append((stamp_ns, gyro, accel))

        # Cap buffer to prevent unbounded growth before first VIO
        if len(self._imu_buf) > 5000:
            self._imu_buf = self._imu_buf[-4000:]

    # ── VIO callback (keyframe trigger) ───────────────────────────────────────

    def _vio_cb(self, msg: Odometry) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        p_meas = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=np.float64)
        q_meas = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=np.float64)
        q_meas /= np.linalg.norm(q_meas)
        R_meas = quat_to_rot(q_meas)

        # Extract covariance from VIO message
        cov_flat = msg.pose.covariance
        if any(c != 0.0 for c in cov_flat):
            pos_var = [cov_flat[0], cov_flat[7], cov_flat[14]]
            ang_var = [cov_flat[21], cov_flat[28], cov_flat[35]]
            cov_6x6 = np.diag(pos_var + ang_var)
        else:
            cov_6x6 = np.diag([
                self._vo_pos_std**2, self._vo_pos_std**2, self._vo_pos_std**2,
                self._vo_ang_std**2, self._vo_ang_std**2, self._vo_ang_std**2,
            ])

        # ── First keyframe: initialize ────────────────────────────────────────
        if self._state == self._UNINIT:
            # Save the FIRST VIO message for gravity computation (its R
            # corresponds to the static period when raw accel was collected)
            if self._first_vio is None:
                self._first_vio = {
                    "stamp_ns": stamp_ns, "p": p_meas.copy(),
                    "R": R_meas.copy(), "q": q_meas.copy(),
                }
            # Wait for enough raw IMU samples for gravity estimation
            if len(self._raw_accel_buf) < self._min_gravity_samples:
                return
            # Init with FIRST VIO pose but use its R for gravity
            fv = self._first_vio
            self._init_from_first_vio(fv["stamp_ns"], fv["p"], fv["R"], fv["q"])
            return

        # ── Subsequent keyframes ──────────────────────────────────────────────
        self._kf_count += 1

        # Relative pose from VIO
        dR_rel = self._prev_vio_R.T @ R_meas
        dp_rel = self._prev_vio_R.T @ (p_meas - self._prev_vio_p)

        # Preintegrate IMU between previous and current keyframe
        self._preintegrate_imu(self._prev_vio_stamp_ns, stamp_ns)

        # Forward-propagate previous optimized state as initial guess
        prev_state = self._graph.latest_state()
        new_state = self._propagate_state(prev_state, stamp_ns)

        # Add to graph
        idx_j = self._graph.add_keyframe(new_state)
        idx_i = idx_j - 1

        self._graph.add_imu_factor(idx_i, idx_j, self._preint)
        self._graph.add_vo_factor(idx_i, idx_j, dR_rel, dp_rel, cov_6x6)

        # Optimize FIRST, then marginalize (Schur complement requires optimized linearization)
        cost = self._graph.optimize()

        if self._graph.num_keyframes > self._window_size:
            self._graph.marginalize_oldest()

        # Publish
        opt_state = self._graph.latest_state()
        self._publish(opt_state, msg.header.stamp)

        # Update biases for next preintegration
        b_a, b_g = self._graph.latest_biases()
        self._preint = ImuPreintegrator(
            b_a=b_a, b_g=b_g,
            sigma_a=self._sigma_a, sigma_g=self._sigma_g,
            sigma_ba=self._sigma_ba, sigma_bg=self._sigma_bg,
            gravity=self._gravity,
        )

        # Store current VIO for next relative pose
        self._prev_vio_p = p_meas.copy()
        self._prev_vio_R = R_meas.copy()
        self._prev_vio_stamp_ns = stamp_ns

        if self._kf_count % 20 == 0:
            self.get_logger().info(
                f"FGO KF#{self._kf_count}: cost={cost:.4f} "
                f"window={self._graph.num_keyframes} "
                f"p={np.round(opt_state.p, 3)}"
            )

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_from_first_vio(self, stamp_ns: int, p: np.ndarray, R: np.ndarray, q: np.ndarray) -> None:
        """Initialize graph from first VIO keyframe."""
        # Estimate gravity from raw IMU buffer
        if len(self._raw_accel_buf) >= 10:
            f_static = np.mean(self._raw_accel_buf, axis=0)
            self._gravity = R @ (-f_static)
            self.get_logger().info(
                f"FGO: estimated gravity: {np.round(self._gravity, 4)} "
                f"(mag={np.linalg.norm(self._gravity):.3f} m/s², "
                f"from {len(self._raw_accel_buf)} samples)"
            )
        else:
            self.get_logger().warn(
                f"FGO: insufficient raw IMU samples ({len(self._raw_accel_buf)}) "
                "for gravity estimation — using config value."
            )
        self._raw_accel_buf = []

        # Create graph
        self._graph = SlidingWindowGraph(
            window_size=self._window_size,
            gravity=self._gravity,
            lm_max_iter=self._lm_max_iter,
            lm_lambda_init=self._lm_lambda_init,
        )

        # Initial state: zero velocity and biases
        init_state = KeyframeState(
            stamp_ns=stamp_ns,
            R=R.copy(),
            p=p.copy(),
            v=np.zeros(3, dtype=np.float64),
            b_a=np.zeros(3, dtype=np.float64),
            b_g=np.zeros(3, dtype=np.float64),
        )
        self._graph.add_keyframe(init_state)

        # Add prior factor on first keyframe
        from vio_pipeline.factor_graph import PriorFactor
        prior_info = np.diag([
            1.0 / self._prior_att_std**2, 1.0 / self._prior_att_std**2, 1.0 / self._prior_att_std**2,
            1.0 / self._prior_pos_std**2, 1.0 / self._prior_pos_std**2, 1.0 / self._prior_pos_std**2,
            1.0 / self._prior_vel_std**2, 1.0 / self._prior_vel_std**2, 1.0 / self._prior_vel_std**2,
            1.0 / self._prior_ba_std**2, 1.0 / self._prior_ba_std**2, 1.0 / self._prior_ba_std**2,
            1.0 / self._prior_bg_std**2, 1.0 / self._prior_bg_std**2, 1.0 / self._prior_bg_std**2,
        ])
        # J = sqrt(info), r0 = 0
        J_prior = np.diag(np.sqrt(np.diag(prior_info)))
        self._graph._prior = PriorFactor(
            node_indices=[0],
            J=J_prior,
            r0=np.zeros(15, dtype=np.float64),
            x_lins=[init_state.copy()],
        )

        # Initialize preintegrator
        self._preint = ImuPreintegrator(
            b_a=np.zeros(3), b_g=np.zeros(3),
            sigma_a=self._sigma_a, sigma_g=self._sigma_g,
            sigma_ba=self._sigma_ba, sigma_bg=self._sigma_bg,
            gravity=self._gravity,
        )

        # Store VIO state
        self._prev_vio_p = p.copy()
        self._prev_vio_R = R.copy()
        self._prev_vio_stamp_ns = stamp_ns

        self._state = self._RUNNING
        self.get_logger().info(f"FGO initialised from first VIO: p={np.round(p, 3)}")

    # ── IMU preintegration between keyframes ──────────────────────────────────

    def _preintegrate_imu(self, t_prev_ns: int, t_curr_ns: int) -> None:
        """Preintegrate buffered IMU samples in [t_prev, t_curr] into self._preint."""
        # Find relevant IMU samples
        samples = []
        remaining = []
        for s in self._imu_buf:
            if s[0] <= t_prev_ns:
                continue
            if s[0] > t_curr_ns:
                remaining.append(s)
                continue
            samples.append(s)

        # Keep unconsumed samples (plus last consumed for next interval boundary)
        self._imu_buf = remaining

        if not samples:
            return

        prev_t_ns = t_prev_ns
        for stamp_ns, gyro, accel in samples:
            dt = (stamp_ns - prev_t_ns) * 1e-9
            if 0.0 < dt < 0.1:
                self._preint.integrate(gyro, accel, dt)
            prev_t_ns = stamp_ns

        # Handle boundary: integrate from last sample to t_curr if needed
        last_t_ns = samples[-1][0]
        if last_t_ns < t_curr_ns:
            dt = (t_curr_ns - last_t_ns) * 1e-9
            if 0.0 < dt < 0.1:
                self._preint.integrate(samples[-1][1], samples[-1][2], dt)

    # ── State propagation for initial guess ───────────────────────────────────

    def _propagate_state(self, prev: KeyframeState, stamp_ns: int) -> KeyframeState:
        """Forward-propagate previous state using preintegration for initial guess."""
        dt = self._preint.delta_t
        g = self._gravity
        Ri = prev.R

        R_new = Ri @ self._preint.delta_R
        v_new = prev.v + g * dt + Ri @ self._preint.delta_v
        p_new = prev.p + prev.v * dt + 0.5 * g * dt * dt + Ri @ self._preint.delta_p

        return KeyframeState(
            stamp_ns=stamp_ns,
            R=R_new,
            p=p_new,
            v=v_new,
            b_a=prev.b_a.copy(),
            b_g=prev.b_g.copy(),
        )

    # ── Publish ───────────────────────────────────────────────────────────────

    def _publish(self, state: KeyframeState, stamp) -> None:
        q = rot_to_quat(state.R)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = state.p[0]
        odom.pose.pose.position.y = state.p[1]
        odom.pose.pose.position.z = state.p[2]
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = state.v[0]
        odom.twist.twist.linear.y = state.v[1]
        odom.twist.twist.linear.z = state.v[2]
        self._pub_odom.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self._pub_pose.publish(pose)

        self._path_msg.header.stamp = stamp
        self._path_msg.poses.append(pose)
        self._pub_path.publish(self._path_msg)


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = FgoBackendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
