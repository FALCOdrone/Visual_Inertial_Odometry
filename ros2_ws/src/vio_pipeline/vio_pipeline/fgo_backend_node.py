#!/usr/bin/env python3
"""
fgo_backend_node.py
===================
Sliding Window Factor Graph Optimization VIO Backend -- Stage A (pose-level).

Fuses raw IMU data (200 Hz) with visual odometry pose estimates (20 Hz)
using on-manifold IMU preintegration and Levenberg-Marquardt optimization
over a sliding window of keyframes.

Subscriptions
-------------
  /imu0             sensor_msgs/Imu        raw IMU data at 200 Hz
  /vio/odometry     nav_msgs/Odometry      visual 6-DOF pose at ~20 Hz

Publications
------------
  /fgo/odometry     nav_msgs/Odometry      optimized fused pose + velocity
  /fgo/pose         geometry_msgs/PoseStamped   convenience pose for RViz

Parameters
----------
  window_size        int    sliding window size (default 10)
  lm_max_iters       int    LM optimization iterations (default 5)
  kf_trans_thresh    float  keyframe selection: translation threshold [m] (default 0.05)
  kf_rot_thresh_deg  float  keyframe selection: rotation threshold [deg] (default 5.0)
  sigma_a            float  accel noise density [m/s^2/sqrt(Hz)] (default 2.0e-3)
  sigma_g            float  gyro noise density [rad/s/sqrt(Hz)] (default 1.6968e-4)
  sigma_ba           float  accel bias walk [m/s^3/sqrt(Hz)] (default 3.0e-3)
  sigma_bg           float  gyro bias walk [rad/s^2/sqrt(Hz)] (default 1.9393e-5)

Architecture
------------
The node runs graph optimization in a background thread to avoid blocking
the high-rate IMU callback.  A threading.Lock protects shared state (the
graph object, current biases, and propagated pose).

Between keyframes, the node propagates a nominal pose at IMU rate for
smooth 200 Hz output, using the latest estimated biases and velocity.
"""

import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from vio_pipeline.imu_preintegrator import (
    _skew, exp_so3, log_so3,
    quat_to_rot, rot_to_quat, quat_mul, ImuPreintegrator
)
from vio_pipeline.factor_graph import (
    KeyframeState, SlidingWindowGraph, _exp_so3_quat
)


# ---- Utility ----------------------------------------------------------------

def _stamp_to_ns(stamp):
    """Convert a ROS2 Time stamp to integer nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _normalize_quat(q):
    """Normalize a quaternion [x, y, z, w] to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


# ---- FGO Backend Node -------------------------------------------------------

class FgoBackendNode(Node):
    """
    Factor Graph Optimization backend for VIO (Stage A: pose-level).

    State machine:
      UNINIT   Waiting for the first VO pose to seed the graph.
      RUNNING  IMU propagation + VO keyframe insertion + graph optimization.
    """

    _UNINIT = "uninit"
    _RUNNING = "running"

    def __init__(self):
        super().__init__("fgo_backend_node")

        # ---- Declare ROS2 parameters ----
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

        self._window_size = int(self.get_parameter("window_size").value)
        self._lm_max_iters = int(self.get_parameter("lm_max_iters").value)
        self._kf_trans_thresh = float(self.get_parameter("kf_trans_thresh").value)
        self._kf_rot_thresh_deg = float(self.get_parameter("kf_rot_thresh_deg").value)
        self._kf_time_thresh = float(self.get_parameter("kf_time_thresh").value)
        self._sigma_a = float(self.get_parameter("sigma_a").value)
        self._sigma_g = float(self.get_parameter("sigma_g").value)
        self._sigma_ba = float(self.get_parameter("sigma_ba").value)
        self._sigma_bg = float(self.get_parameter("sigma_bg").value)
        self._max_dt = float(self.get_parameter("max_dt").value)

        self.get_logger().info(
            f"FGO params: window={self._window_size} iters={self._lm_max_iters} "
            f"kf_trans={self._kf_trans_thresh}m kf_rot={self._kf_rot_thresh_deg}deg "
            f"sigma_a={self._sigma_a} sigma_g={self._sigma_g} "
            f"sigma_ba={self._sigma_ba} sigma_bg={self._sigma_bg}")

        # ---- Constants ----
        self._gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Camera intrinsics (cam0 after rectification -- EuRoC defaults)
        cam_K = np.array([
            [458.654, 0.0, 367.215],
            [0.0, 457.296, 248.375],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Body <- cam0 transform (identity as default; the actual T_BS0 from
        # EuRoC is close to identity after adjusting for rectification)
        T_b_c0 = np.eye(4, dtype=np.float64)

        # ---- Factor graph ----
        self._graph = SlidingWindowGraph(
            cam_K=cam_K,
            T_b_c0=T_b_c0,
            gravity=self._gravity,
            window_size=self._window_size)
        self._graph._sigma_ba = self._sigma_ba
        self._graph._sigma_bg = self._sigma_bg

        # ---- State ----
        self._state = self._UNINIT
        self._lock = threading.Lock()

        # Current bias estimates (updated after each optimization)
        self._b_a = np.zeros(3, dtype=np.float64)
        self._b_g = np.zeros(3, dtype=np.float64)

        # Propagated nominal pose (updated at IMU rate for smooth output)
        self._prop_p = np.zeros(3, dtype=np.float64)
        self._prop_v = np.zeros(3, dtype=np.float64)
        self._prop_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        # IMU preintegrator for current interval (between keyframes)
        self._preintegrator = None

        # Last keyframe state (for keyframe selection)
        self._last_kf_p = np.zeros(3, dtype=np.float64)
        self._last_kf_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._last_kf_stamp_ns = 0

        # Timestamp tracking
        self._last_imu_stamp_ns = None

        # Raw IMU buffer for gravity estimation during static init
        self._raw_accel_buf = []
        self._RAW_BUF_MAX = 2000

        # Optimization thread handle
        self._opt_thread = None

        # Counter for logging
        self._kf_count = 0
        self._vo_count = 0

        # ---- QoS ----
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50)

        # ---- Publishers ----
        self._pub_odom = self.create_publisher(Odometry, "/fgo/odometry", qos_be)
        self._pub_pose = self.create_publisher(PoseStamped, "/fgo/pose", qos_be)

        # ---- Subscribers ----
        # Use bias-corrected + filtered IMU (matches ESKF behaviour).
        # imu_processing_node removes the static gyro/accel bias and applies LPF,
        # so our b_a=0, b_g=0 initialisation is a valid linearisation point.
        # Raw /imu0 is still buffered for gravity estimation before init.
        self.create_subscription(Imu, "/imu0", self._raw_imu_cb, qos_be)
        self.create_subscription(Imu, "/imu/processed", self._imu_cb, qos_be)
        # Visual odometry from frontend (pose_estimation_node publishes here)
        self.create_subscription(Odometry, "/vio/odometry", self._vo_cb, 10)

        self.get_logger().info(
            "FgoBackendNode ready -- waiting for first VO measurement to initialize.")

    # ---- IMU callbacks -------------------------------------------------------

    def _raw_imu_cb(self, msg):
        """Buffer raw /imu0 accelerometer for gravity estimation before init."""
        if self._state != self._UNINIT:
            return
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z], dtype=np.float64)
        if len(self._raw_accel_buf) < self._RAW_BUF_MAX:
            self._raw_accel_buf.append(accel.copy())

    def _imu_cb(self, msg):
        """
        Process bias-corrected + filtered IMU from /imu/processed at 200 Hz.

        imu_processing_node has already removed the static gyro/accel bias,
        so the FGO's b_a=0, b_g=0 initialisation is a valid starting point.
        The FGO optimizer refines the residual bias during the sliding window.
        """
        if self._state == self._UNINIT:
            return

        stamp_ns = _stamp_to_ns(msg.header.stamp)

        omega = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z], dtype=np.float64)
        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z], dtype=np.float64)

        # Compute dt
        if self._last_imu_stamp_ns is None:
            self._last_imu_stamp_ns = stamp_ns
            return
        dt = (stamp_ns - self._last_imu_stamp_ns) * 1e-9
        self._last_imu_stamp_ns = stamp_ns

        if dt <= 0.0 or dt > self._max_dt:
            return

        with self._lock:
            b_a = self._b_a.copy()
            b_g = self._b_g.copy()

            # Feed into preintegrator (b_a/b_g are residual bias after
            # imu_processing_node's initial removal)
            if self._preintegrator is not None:
                omega_corr = omega - b_g
                accel_corr = accel - b_a
                self._preintegrator.integrate(
                    omega_corr, accel_corr, dt, omega, accel)

            # Propagate nominal pose for smooth high-rate output
            R = quat_to_rot(self._prop_q)
            f_corr = accel - b_a
            w_corr = omega - b_g
            a_world = R @ f_corr + self._gravity
            self._prop_p += self._prop_v * dt + 0.5 * a_world * dt * dt
            self._prop_v += a_world * dt
            dq = _exp_so3_quat(w_corr * dt)
            self._prop_q = _normalize_quat(quat_mul(self._prop_q, dq))

        # Publish propagated pose
        self._publish(msg.header)

    # ---- VO callback ---------------------------------------------------------

    def _vo_cb(self, msg):
        """
        Process a visual odometry pose estimate (~20 Hz).

        On the first message: initialize the graph with the VO pose.
        On subsequent messages: check keyframe criteria and, if triggered,
        add a new keyframe to the graph and run optimization.
        """
        p_meas = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z], dtype=np.float64)
        q_meas = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w], dtype=np.float64)
        q_meas = _normalize_quat(q_meas)
        stamp_ns = _stamp_to_ns(msg.header.stamp)

        self._vo_count += 1

        if self._state == self._UNINIT:
            self._initialize(p_meas, q_meas, stamp_ns)
            return

        # ---- Keyframe selection ----
        if not self._is_keyframe(p_meas, q_meas, stamp_ns):
            return

        self._kf_count += 1

        with self._lock:
            # Create new keyframe state
            # Use VO pose for position and orientation,
            # propagated velocity, and current bias estimates
            kf_state = KeyframeState(
                p=p_meas,
                v=self._prop_v.copy(),
                q=q_meas,
                b_a=self._b_a.copy(),
                b_g=self._b_g.copy(),
                stamp_ns=stamp_ns)

            # Add keyframe with the preintegrated IMU measurements
            preint = self._preintegrator
            kf_id = self._graph.add_keyframe(kf_state, preint)

            # Anchor the keyframe to the VO pose measurement (Stage A)
            # sigma_trans: VO position noise ~5 cm, sigma_rot: ~5 deg
            self._graph.add_pose_factor(kf_id, p_meas, q_meas,
                                        sigma_trans=0.05,
                                        sigma_rot=np.radians(5.0))

            # Start new preintegrator for the next interval
            self._preintegrator = ImuPreintegrator(
                self._b_a, self._b_g,
                self._sigma_a, self._sigma_g,
                self._sigma_ba, self._sigma_bg)

            # Update last keyframe reference
            self._last_kf_p = p_meas.copy()
            self._last_kf_q = q_meas.copy()
            self._last_kf_stamp_ns = stamp_ns

        # Run optimization in background thread
        if self._opt_thread is None or not self._opt_thread.is_alive():
            self._opt_thread = threading.Thread(target=self._run_optimization, daemon=True)
            self._opt_thread.start()

    def _initialize(self, p, q, stamp_ns):
        """
        Initialize the FGO backend from the first VO pose.

        Seeds the graph with the initial keyframe and estimates gravity
        from the buffered raw IMU data (if available).
        """
        # Estimate gravity in the VIO world frame from static accel buffer
        if len(self._raw_accel_buf) >= 10:
            R_init = quat_to_rot(q)
            f_static = np.mean(self._raw_accel_buf, axis=0)
            self._gravity = R_init @ (-f_static)
            self.get_logger().info(
                f"FGO: estimated gravity in VIO world frame: "
                f"{np.round(self._gravity, 4)} "
                f"(mag={np.linalg.norm(self._gravity):.3f} m/s^2, "
                f"from {len(self._raw_accel_buf)} raw samples)")
        else:
            self.get_logger().warn(
                f"FGO: insufficient raw IMU samples ({len(self._raw_accel_buf)}) "
                "for gravity estimation -- using default [0, 0, -9.81].")
        self._raw_accel_buf = []  # free memory

        # Update graph gravity
        self._graph.gravity = self._gravity.copy()

        # Create initial keyframe
        kf_state = KeyframeState(
            p=p, v=np.zeros(3), q=q,
            b_a=np.zeros(3), b_g=np.zeros(3),
            stamp_ns=stamp_ns)

        with self._lock:
            kf_id0 = self._graph.add_keyframe(kf_state, preintegrator=None)
            # Strong prior on first keyframe (origin anchor)
            self._graph.add_pose_factor(kf_id0, p, q,
                                        sigma_trans=0.001,
                                        sigma_rot=np.radians(0.5))
            # Anchor initial velocity to zero (robot starts from rest).
            # Velocity has no absolute factor from IMU alone — this prevents
            # the velocity null-space from causing numerical explosion.
            self._graph.add_velocity_prior(kf_id0, np.zeros(3), sigma_v=0.3)

            # Initialize propagated state
            self._prop_p = p.copy()
            self._prop_v = np.zeros(3, dtype=np.float64)
            self._prop_q = q.copy()

            # Initialize bias estimates
            self._b_a = np.zeros(3, dtype=np.float64)
            self._b_g = np.zeros(3, dtype=np.float64)

            # Start preintegrator for the interval after this keyframe
            self._preintegrator = ImuPreintegrator(
                self._b_a, self._b_g,
                self._sigma_a, self._sigma_g,
                self._sigma_ba, self._sigma_bg)

            # Record as last keyframe
            self._last_kf_p = p.copy()
            self._last_kf_q = q.copy()
            self._last_kf_stamp_ns = stamp_ns
            self._last_imu_stamp_ns = None  # will be set on next IMU msg

        self._state = self._RUNNING
        self._kf_count = 1
        self.get_logger().info(
            f"FGO initialized from first VO pose: p={np.round(p, 3)}")

    def _is_keyframe(self, p, q, stamp_ns):
        """
        Decide whether the current VO pose qualifies as a new keyframe.

        Criteria (any one triggers):
          1. Translation from last keyframe > kf_trans_thresh
          2. Rotation from last keyframe > kf_rot_thresh_deg
          3. Time since last keyframe > kf_time_thresh
        """
        # Translation check
        dp = np.linalg.norm(p - self._last_kf_p)
        if dp > self._kf_trans_thresh:
            return True

        # Rotation check
        R_prev = quat_to_rot(self._last_kf_q)
        R_curr = quat_to_rot(q)
        dR = R_prev.T @ R_curr
        angle_rad = np.linalg.norm(log_so3(dR))
        if np.degrees(angle_rad) > self._kf_rot_thresh_deg:
            return True

        # Time check
        dt = (stamp_ns - self._last_kf_stamp_ns) * 1e-9
        if dt > self._kf_time_thresh:
            return True

        return False

    def _run_optimization(self):
        """
        Run the factor graph optimization in a background thread.

        After optimization, extract the updated biases and velocity from
        the latest keyframe and update the propagated state.
        """
        with self._lock:
            if self._graph.num_keyframes() < 2:
                return

            cost = self._graph.optimize(max_iters=self._lm_max_iters)

            # Extract updated state from the latest keyframe
            kf_id, latest_kf = self._graph.latest_keyframe()
            if latest_kf is None:
                return

            # Update biases from optimized keyframe
            self._b_a = latest_kf.b_a.copy()
            self._b_g = latest_kf.b_g.copy()

            # Re-anchor propagated state to the optimized keyframe, then
            # re-apply the IMU delta accumulated since that keyframe.
            preint = self._preintegrator
            dp, dv, dR = preint.bias_corrected_measurement(
                latest_kf.b_a, latest_kf.b_g)
            dt = preint.dt_total
            g = self._gravity
            R_kf = quat_to_rot(latest_kf.q)
            self._prop_p = latest_kf.p + latest_kf.v * dt + 0.5 * g * dt**2 + R_kf @ dp
            self._prop_v = latest_kf.v + g * dt + R_kf @ dv
            self._prop_q = _normalize_quat(rot_to_quat(R_kf @ dR))

        if self._kf_count % 10 == 0:
            self.get_logger().info(
                f"FGO opt done: kf={self._kf_count} cost={cost:.4f} "
                f"b_a={np.round(self._b_a, 5)} b_g={np.round(self._b_g, 7)}")

    # ---- Publishing ----------------------------------------------------------

    def _publish(self, header):
        """Publish the current propagated pose as Odometry and PoseStamped."""
        stamp = header.stamp

        with self._lock:
            p = self._prop_p.copy()
            q = self._prop_q.copy()
            v = self._prop_v.copy()

        # Odometry message
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

        # PoseStamped (convenience for RViz)
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self._pub_pose.publish(pose)


# ---- Entry point -------------------------------------------------------------

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
