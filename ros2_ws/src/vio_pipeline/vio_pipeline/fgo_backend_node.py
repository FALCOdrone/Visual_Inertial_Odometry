#!/usr/bin/env python3
"""
fgo_backend_node.py
===================
ROS2 node implementing a Sliding-Window Factor Graph (FGO) VIO backend.

Subscribes
----------
  /imu0            sensor_msgs/Imu   Raw IMU for gravity estimation (UNINIT only)
  /imu/processed   sensor_msgs/Imu   Bias-corrected IMU @ ~200 Hz  (RUNNING)
  /vio/odometry    nav_msgs/Odometry Visual-odometry pose @ ~20 Hz

Publishes
---------
  /fgo/odometry    nav_msgs/Odometry  Fused pose + velocity (map frame)
  /fgo/pose        geometry_msgs/PoseStamped  Convenience pose
  /fgo/path        nav_msgs/Path      Accumulated trajectory

Parameters
----------
  window_size      int    Maximum keyframes in sliding window (default 10)
  lm_max_iters     int    LM iterations per VO update (default 5)
  kf_trans_thresh  float  Keyframe translation threshold [m]   (default 0.05)
  kf_rot_thresh_deg float Keyframe rotation threshold [deg]    (default 5.0)
  kf_time_thresh   float  Keyframe time threshold [s]          (default 0.25)
  sigma_a          float  Accel noise density [m/s^2/sqrt(Hz)] (default 2.0e-3)
  sigma_g          float  Gyro  noise density [rad/s/sqrt(Hz)] (default 1.6968e-4)
  sigma_ba         float  Accel bias walk [m/s^3/sqrt(Hz)]     (default 3.0e-3)
  sigma_bg         float  Gyro  bias walk [rad/s^2/sqrt(Hz)]   (default 1.9393e-5)
  max_dt           float  Maximum allowed IMU dt [s]            (default 0.1)

State machine
-------------
  UNINIT   Waiting for the first VO measurement to initialize.
  RUNNING  IMU propagation + VO keyframe insertion + background optimization.

Notes on thread safety
----------------------
All shared state (propagated pose, preintegrator, biases, graph) is protected
by a single threading.Lock.  The optimization thread acquires and releases the
lock for the duration of the graph.optimize() call.  IMU callbacks may queue
during this time (short — typically < 1 ms per iteration for a 10-KF window).
"""

import math
import threading
from math import radians

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

from vio_pipeline.imu_preintegrator import (
    ImuPreintegrator,
    exp_so3,
    log_so3,
    quat_to_rot,
    rot_to_quat,
    quat_mul,
)
from vio_pipeline.factor_graph import (
    KeyframeState,
    SlidingWindowGraph,
    _exp_so3_quat,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _stamp_to_ns(stamp) -> int:
    """Convert a ROS2 header stamp to nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion; return identity [0,0,0,1] if norm is near zero."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


# ──────────────────────────────────────────────────────────────────────────────
#  FgoBackendNode
# ──────────────────────────────────────────────────────────────────────────────


class FgoBackendNode(Node):
    """Sliding-Window Factor Graph backend node.

    The node runs two concurrent activities:
      1. High-rate IMU callback: propagates the nominal pose at ~200 Hz and
         accumulates preintegrated measurements.
      2. VO callback: decides whether to insert a keyframe, then spawns a
         background thread to run Levenberg-Marquardt optimization.

    After each optimization, the propagated state is re-anchored to the latest
    optimized keyframe using the accumulated preintegration since that keyframe.
    """

    _UNINIT = "uninit"
    _RUNNING = "running"

    def __init__(self) -> None:
        super().__init__("fgo_backend_node")

        # ── Declare parameters ─────────────────────────────────────────────
        self.declare_parameter("window_size", 10)
        self.declare_parameter("lm_max_iters", 5)
        self.declare_parameter("kf_trans_thresh", 0.05)
        self.declare_parameter("kf_rot_thresh_deg", 5.0)
        self.declare_parameter("kf_time_thresh", 0.25)
        self.declare_parameter("sigma_a", 2.0e-3)
        self.declare_parameter("sigma_g", 1.6968e-4)
        self.declare_parameter("sigma_ba", 3.0e-3)
        self.declare_parameter("sigma_bg", 1.9393e-5)
        self.declare_parameter("max_dt", 0.1)

        # ── Read parameters ────────────────────────────────────────────────
        window_size       = int(self.get_parameter("window_size").value)
        self._lm_max_iters = int(self.get_parameter("lm_max_iters").value)
        self._kf_trans_thresh = float(self.get_parameter("kf_trans_thresh").value)
        self._kf_rot_thresh_rad = radians(
            float(self.get_parameter("kf_rot_thresh_deg").value)
        )
        self._kf_time_thresh = float(self.get_parameter("kf_time_thresh").value)
        self._sigma_a  = float(self.get_parameter("sigma_a").value)
        self._sigma_g  = float(self.get_parameter("sigma_g").value)
        self._sigma_ba = float(self.get_parameter("sigma_ba").value)
        self._sigma_bg = float(self.get_parameter("sigma_bg").value)
        self._max_dt   = float(self.get_parameter("max_dt").value)

        self.get_logger().info(
            "FGO params: window=%d  lm_iters=%d  "
            "kf_trans=%.3f m  kf_rot=%.1f deg  kf_time=%.2f s  "
            "sigma_a=%.2e  sigma_g=%.2e  sigma_ba=%.2e  sigma_bg=%.2e  "
            "max_dt=%.2f s"
            % (
                window_size,
                self._lm_max_iters,
                self._kf_trans_thresh,
                float(self.get_parameter("kf_rot_thresh_deg").value),
                self._kf_time_thresh,
                self._sigma_a,
                self._sigma_g,
                self._sigma_ba,
                self._sigma_bg,
                self._max_dt,
            )
        )

        # ── Factor graph ───────────────────────────────────────────────────
        self._graph = SlidingWindowGraph(
            gravity=np.array([0.0, 0.0, -9.81], dtype=np.float64),
            window_size=window_size,
        )
        self._graph._sigma_ba = self._sigma_ba
        self._graph._sigma_bg = self._sigma_bg

        # ── State machine ──────────────────────────────────────────────────
        self._state: str = self._UNINIT
        self._lock = threading.Lock()
        self._opt_thread: threading.Thread | None = None
        self._opt_running: bool = False

        # ── Raw IMU buffer for gravity estimation ──────────────────────────
        # Collect raw /imu0 accel during the static init window.
        # Capped at 2000 samples (10 s @ 200 Hz).
        self._raw_accel_buf: list[np.ndarray] = []
        self._RAW_BUF_MAX = 2000

        # ── Propagated nominal state ───────────────────────────────────────
        # Updated every IMU step; re-anchored after each optimization.
        self._prop_p: np.ndarray = np.zeros(3, dtype=np.float64)
        self._prop_v: np.ndarray = np.zeros(3, dtype=np.float64)
        self._prop_q: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        # ── Current IMU biases (updated from optimizer) ────────────────────
        self._b_a: np.ndarray = np.zeros(3, dtype=np.float64)
        self._b_g: np.ndarray = np.zeros(3, dtype=np.float64)

        # ── Preintegrator (between last KF and "now") ──────────────────────
        self._preint: ImuPreintegrator | None = None

        # ── Last IMU timestamp (ns) ────────────────────────────────────────
        self._last_imu_stamp_ns: int | None = None

        # ── Last keyframe state (for KF selection heuristic) ──────────────
        self._last_kf_p: np.ndarray = np.zeros(3, dtype=np.float64)
        self._last_kf_q: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._last_kf_stamp_ns: int = 0

        # ── Keyframe count for periodic logging ───────────────────────────
        self._kf_count: int = 0

        # ── Path message (accumulated trajectory) ─────────────────────────
        self._path_msg = Path()
        self._path_msg.header.frame_id = "map"

        # ── QoS ───────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # ── Publishers ─────────────────────────────────────────────────────
        self._pub_odom = self.create_publisher(Odometry,       "/fgo/odometry", qos_be)
        self._pub_pose = self.create_publisher(PoseStamped,    "/fgo/pose",     qos_be)
        self._pub_path = self.create_publisher(Path,           "/fgo/path",     qos_be)

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(Imu,      "/imu0",          self._raw_imu_cb, qos_be)
        self.create_subscription(Imu,      "/imu/processed", self._imu_cb,     qos_be)
        self.create_subscription(Odometry, "/vio/odometry",  self._vo_cb,      10)

        self.get_logger().info(
            "FgoBackendNode ready — waiting for first VO measurement to initialize."
        )

    # ── Raw IMU (gravity estimation) ───────────────────────────────────────────

    def _raw_imu_cb(self, msg: Imu) -> None:
        """Buffer raw accelerometer readings for gravity estimation during UNINIT."""
        if self._state != self._UNINIT or len(self._raw_accel_buf) >= self._RAW_BUF_MAX:
            return
        self._raw_accel_buf.append(
            np.array(
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ],
                dtype=np.float64,
            )
        )

    # ── Initialization ─────────────────────────────────────────────────────────

    def _initialize(self, p: np.ndarray, q: np.ndarray, stamp_ns: int) -> None:
        """Initialize the graph with the first VO pose.

        Estimates the gravity vector from the buffered raw accelerometer data,
        then seeds the graph with the first keyframe.

        Parameters
        ----------
        p        : (3,)  First VO position
        q        : (4,)  First VO orientation [x,y,z,w]
        stamp_ns : int   Timestamp in nanoseconds
        """
        # ── Estimate gravity in VIO world frame ───────────────────────────
        if len(self._raw_accel_buf) >= 10:
            R_init = quat_to_rot(q)
            f_static = np.mean(self._raw_accel_buf, axis=0)
            # f_static = -g_body, so g_world = R @ (-f_static)
            gravity = R_init @ (-f_static)
            self._graph.gravity = gravity
            self.get_logger().info(
                "FGO: estimated gravity in VIO world frame: %s  "
                "(mag=%.3f m/s², from %d raw samples)"
                % (
                    np.round(gravity, 4),
                    float(np.linalg.norm(gravity)),
                    len(self._raw_accel_buf),
                )
            )
        else:
            self.get_logger().warn(
                "FGO: insufficient raw IMU samples (%d) for gravity estimation "
                "— using default [0, 0, -9.81]." % len(self._raw_accel_buf)
            )
        self._raw_accel_buf = []  # free memory

        # ── Create initial keyframe ────────────────────────────────────────
        kf_state = KeyframeState(
            p=p,
            v=np.zeros(3, dtype=np.float64),
            q=q,
            b_a=np.zeros(3, dtype=np.float64),
            b_g=np.zeros(3, dtype=np.float64),
            stamp_ns=stamp_ns,
        )
        kf_id0 = self._graph.add_keyframe(kf_state, preintegrator=None)

        # Tight pose anchor and zero-velocity prior for the first KF
        self._graph.add_pose_factor(
            kf_id0, p, q, sigma_trans=0.001, sigma_rot=radians(0.5)
        )
        self._graph.add_velocity_prior(
            kf_id0, np.zeros(3, dtype=np.float64), sigma_v=0.3
        )

        # ── Initialize propagated state ────────────────────────────────────
        self._prop_p = p.copy()
        self._prop_v = np.zeros(3, dtype=np.float64)
        self._prop_q = q.copy()

        # ── Initialize biases ─────────────────────────────────────────────
        self._b_a = np.zeros(3, dtype=np.float64)
        self._b_g = np.zeros(3, dtype=np.float64)

        # ── Initialize preintegrator for the interval since this KF ───────
        self._preint = ImuPreintegrator(
            b_a=self._b_a,
            b_g=self._b_g,
            sigma_a=self._sigma_a,
            sigma_g=self._sigma_g,
            sigma_ba=self._sigma_ba,
            sigma_bg=self._sigma_bg,
        )

        # ── Update KF selection bookkeeping ───────────────────────────────
        self._last_kf_p = p.copy()
        self._last_kf_q = q.copy()
        self._last_kf_stamp_ns = stamp_ns
        self._last_imu_stamp_ns = stamp_ns

        self._kf_count = 1
        self._state = self._RUNNING
        self.get_logger().info(
            "FGO initialized from first VO pose: p=%s" % np.round(p, 3)
        )

    # ── IMU propagation ────────────────────────────────────────────────────────

    def _imu_cb(self, msg: Imu) -> None:
        """High-rate IMU callback: propagate nominal state and preintegrate."""
        if self._state != self._RUNNING:
            return

        stamp_ns = _stamp_to_ns(msg.header.stamp)

        if self._last_imu_stamp_ns is None:
            self._last_imu_stamp_ns = stamp_ns
            return

        dt = (stamp_ns - self._last_imu_stamp_ns) * 1e-9
        self._last_imu_stamp_ns = stamp_ns

        if dt <= 0.0 or dt > self._max_dt:
            return

        omega = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=np.float64,
        )
        accel = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=np.float64,
        )

        # Guard against NaN/Inf in sensor readings
        if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(accel)):
            return

        with self._lock:
            if self._preint is None:
                return

            omega_corr = omega - self._b_g
            accel_corr = accel - self._b_a

            # Preintegrate this sample (stores raw for potential re-integration)
            self._preint.integrate(omega_corr, accel_corr, dt, omega, accel)

            # Propagate nominal pose in world frame
            R = quat_to_rot(self._prop_q)
            a_world = R @ accel_corr + self._graph.gravity

            self._prop_p = self._prop_p + self._prop_v * dt + 0.5 * a_world * dt * dt
            self._prop_v = self._prop_v + a_world * dt
            self._prop_q = _normalize_quat(
                quat_mul(self._prop_q, _exp_so3_quat(omega_corr * dt))
            )

        self._publish(msg.header)

    # ── VO update and keyframe insertion ──────────────────────────────────────

    def _vo_cb(self, msg: Odometry) -> None:
        """VO odometry callback: initialize or insert a new keyframe."""
        p_meas = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        q_meas = _normalize_quat(
            np.array(
                [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ],
                dtype=np.float64,
            )
        )

        stamp_ns = _stamp_to_ns(msg.header.stamp)

        # Guard against non-finite measurements
        if not np.all(np.isfinite(p_meas)) or not np.all(np.isfinite(q_meas)):
            self.get_logger().warn("FGO: received non-finite VO measurement — skipping.")
            return

        # ── Initialization ─────────────────────────────────────────────────
        if self._state == self._UNINIT:
            self._initialize(p_meas, q_meas, stamp_ns)
            return

        # ── Keyframe selection heuristic ───────────────────────────────────
        if not self._is_keyframe(p_meas, q_meas, stamp_ns):
            return

        with self._lock:
            if self._preint is None:
                return

            # Build keyframe state from VO pose + current propagated velocity
            kf_state = KeyframeState(
                p=p_meas,
                v=self._prop_v.copy(),
                q=q_meas,
                b_a=self._b_a.copy(),
                b_g=self._b_g.copy(),
                stamp_ns=stamp_ns,
            )

            # Insert into graph with IMU factor connecting to previous KF
            kf_id = self._graph.add_keyframe(kf_state, self._preint)

            # Add a VO pose anchor (loose — VO has noise)
            self._graph.add_pose_factor(
                kf_id, p_meas, q_meas, sigma_trans=0.05, sigma_rot=radians(5.0)
            )

            # Reset preintegrator for the next interval
            self._preint = ImuPreintegrator(
                b_a=self._b_a.copy(),
                b_g=self._b_g.copy(),
                sigma_a=self._sigma_a,
                sigma_g=self._sigma_g,
                sigma_ba=self._sigma_ba,
                sigma_bg=self._sigma_bg,
            )

            # Update KF selection bookkeeping
            self._last_kf_p = p_meas.copy()
            self._last_kf_q = q_meas.copy()
            self._last_kf_stamp_ns = stamp_ns
            self._kf_count += 1

            if self._kf_count % 10 == 0:
                self.get_logger().info(
                    "FGO: %d keyframes inserted  "
                    "b_a=%s  b_g=%s"
                    % (
                        self._kf_count,
                        np.round(self._b_a, 5),
                        np.round(self._b_g, 7),
                    )
                )

        # ── Spawn background optimization if not already running ───────────
        self._maybe_start_opt()

    def _is_keyframe(
        self, p: np.ndarray, q: np.ndarray, stamp_ns: int
    ) -> bool:
        """Return True if this VO measurement should trigger a new keyframe.

        Criteria (any one sufficient):
          - Translation from last KF > kf_trans_thresh
          - Rotation from last KF   > kf_rot_thresh_rad
          - Time since last KF      > kf_time_thresh
        """
        dt_s = (stamp_ns - self._last_kf_stamp_ns) * 1e-9
        if dt_s >= self._kf_time_thresh:
            return True

        d_trans = float(np.linalg.norm(p - self._last_kf_p))
        if d_trans >= self._kf_trans_thresh:
            return True

        R_last = quat_to_rot(self._last_kf_q)
        R_curr = quat_to_rot(q)
        d_rot = float(np.linalg.norm(log_so3(R_last.T @ R_curr)))
        if d_rot >= self._kf_rot_thresh_rad:
            return True

        return False

    # ── Background optimization ────────────────────────────────────────────────

    def _maybe_start_opt(self) -> None:
        """Spawn an optimization thread if none is running."""
        if self._opt_running:
            return
        self._opt_running = True
        self._opt_thread = threading.Thread(
            target=self._run_optimization, daemon=True
        )
        self._opt_thread.start()

    def _run_optimization(self) -> None:
        """Run LM optimization without blocking IMU callbacks.

        Approach:
          1. Snapshot the graph under a brief lock.
          2. Optimize the snapshot with the lock released — IMU callbacks
             continue to run freely during the (potentially slow) LM loop.
          3. Write the optimized keyframe states + updated biases back under
             a brief lock.  The propagated state (_prop_p/v/q) is deliberately
             NOT re-anchored so that the published trajectory remains smooth
             and continuous (no discontinuous jumps after each optimization).
        """
        import copy  # local import — avoids top-level dependency
        try:
            # ── Step 1: snapshot ──────────────────────────────────────────
            with self._lock:
                if self._preint is None:
                    return
                graph_snapshot = copy.deepcopy(self._graph)

            # ── Step 2: optimize snapshot (lock is FREE) ──────────────────
            cost = graph_snapshot.optimize(max_iters=self._lm_max_iters)

            # ── Step 3: apply results ─────────────────────────────────────
            with self._lock:
                # Copy optimized KF states back to the live graph, matched by ID.
                for i, kf_id in enumerate(self._graph._kf_ids):
                    for j, snap_id in enumerate(graph_snapshot._kf_ids):
                        if snap_id == kf_id:
                            self._graph._keyframes[i] = (
                                graph_snapshot._keyframes[j].copy()
                            )
                            break

                # Propagate updated marginalization prior.
                self._graph._prior_H = graph_snapshot._prior_H
                self._graph._prior_b = graph_snapshot._prior_b
                self._graph._prior_kf_ids = list(graph_snapshot._prior_kf_ids)

                # Update biases and re-anchor from the latest OPTIMIZED keyframe
                # (use the snapshot, not the live graph, to avoid reading a
                # newly-inserted, un-optimized keyframe added during optimization).
                _, latest_kf = graph_snapshot.latest_keyframe()
                if latest_kf is None:
                    return
                self._b_a = latest_kf.b_a.copy()
                self._b_g = latest_kf.b_g.copy()

                # Re-anchor the propagated state to the optimized keyframe plus
                # the preintegrated delta since that keyframe.  This propagates
                # FGO position/orientation corrections to the published trajectory.
                # Corrections are now small (COV_FLOOR fix balanced IMU vs VO
                # information), so jumps are negligible.
                dp, dv, dR = self._preint.bias_corrected_measurement(
                    latest_kf.b_a, latest_kf.b_g
                )
                dt = self._preint.dt_total
                R_kf = quat_to_rot(latest_kf.q)
                g = self._graph.gravity

                new_p = latest_kf.p + latest_kf.v * dt + 0.5 * g * dt * dt + R_kf @ dp
                new_v = latest_kf.v + g * dt + R_kf @ dv
                new_q = _normalize_quat(rot_to_quat(R_kf @ dR))

                if (
                    np.all(np.isfinite(new_p))
                    and np.all(np.isfinite(new_v))
                    and np.all(np.isfinite(new_q))
                ):
                    self._prop_p = new_p
                    self._prop_v = new_v
                    self._prop_q = new_q
                else:
                    self.get_logger().warn(
                        "FGO: non-finite re-anchored state — keeping current propagation."
                    )

                self.get_logger().debug(
                    "FGO opt: cost=%.4f  n_kf=%d  b_a=%s  b_g=%s"
                    % (
                        cost,
                        self._graph.num_keyframes(),
                        np.round(self._b_a, 5),
                        np.round(self._b_g, 7),
                    )
                )
        finally:
            self._opt_running = False

    # ── Publishing ─────────────────────────────────────────────────────────────

    def _publish(self, header) -> None:
        """Publish the current propagated state as Odometry, PoseStamped, Path."""
        with self._lock:
            p = self._prop_p.copy()
            q = self._prop_q.copy()
            v = self._prop_v.copy()

        stamp = header.stamp

        # ── Odometry ───────────────────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(p[0])
        odom.pose.pose.position.y = float(p[1])
        odom.pose.pose.position.z = float(p[2])
        odom.pose.pose.orientation.x = float(q[0])
        odom.pose.pose.orientation.y = float(q[1])
        odom.pose.pose.orientation.z = float(q[2])
        odom.pose.pose.orientation.w = float(q[3])
        odom.twist.twist.linear.x = float(v[0])
        odom.twist.twist.linear.y = float(v[1])
        odom.twist.twist.linear.z = float(v[2])
        self._pub_odom.publish(odom)

        # ── PoseStamped ────────────────────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header = odom.header
        pose_msg.pose = odom.pose.pose
        self._pub_pose.publish(pose_msg)

        # ── Path ───────────────────────────────────────────────────────────
        self._path_msg.header.stamp = stamp
        self._path_msg.poses.append(pose_msg)
        self._pub_path.publish(self._path_msg)


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main(args=None) -> None:
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
