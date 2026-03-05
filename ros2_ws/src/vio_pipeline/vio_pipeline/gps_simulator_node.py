#!/usr/bin/env python3
"""
gps_simulator_node.py
=====================
Simulates a GPS receiver by sampling /gt_pub/pose (ground-truth PoseStamped
in ENU map frame) and corrupting it with realistic GPS error sources.

Error model
-----------
  position_gps = p_true + bias + white_noise + multipath

  white_noise  ~ N(0, diag(σ_h², σ_h², σ_v²))            iid per sample
  bias         Gauss-Markov:  ḃ = -b/τ + q_b              (3-D, in ENU)
  multipath    sporadic:  N(0, (k·σ_h)²) with probability p_multi per sample

The corrupted ENU position is then converted to Latitude/Longitude/Altitude
given a configurable WGS-84 reference origin.

Topics
------
  Subscriptions:
    /gt_pub/pose          geometry_msgs/PoseStamped   ground-truth pose (map/ENU)

  Publications:
    /gps/fix              sensor_msgs/NavSatFix        GPS fix  (LLA + covariance)
    /gps/enu              geometry_msgs/PointStamped   corrupted ENU position (debug)

Parameters
----------
  update_rate_hz      float   GPS publishing rate                [default: 5.0]
  ref_lat_deg         float   WGS-84 reference latitude (deg)   [default: 47.3977419]
  ref_lon_deg         float   WGS-84 reference longitude (deg)  [default: 8.5455938]
  ref_alt_m           float   WGS-84 reference altitude (m)     [default: 486.0]
  noise_h_m           float   Horizontal position 1-σ (m)       [default: 1.5]
  noise_v_m           float   Vertical position 1-σ (m)         [default: 3.0]
  bias_time_const_s   float   Gauss-Markov bias time constant   [default: 60.0]
  bias_walk_h_m_s     float   Horizontal bias walk σ (m/√s)     [default: 0.1]
  bias_walk_v_m_s     float   Vertical bias walk σ (m/√s)       [default: 0.15]
  multipath_prob      float   Per-sample multipath probability   [default: 0.02]
  multipath_scale     float   Multipath error scale factor (×σ_h)[default: 4.0]
  outage_prob_per_s   float   Probability of outage per second   [default: 0.0]
  outage_duration_s   float   Duration of each outage (s)        [default: 3.0]
  seed                int     RNG seed (-1 = random)             [default: -1]
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import NavSatFix, NavSatStatus


# ── WGS-84 constants ──────────────────────────────────────────────────────────

_WGS84_A = 6_378_137.0          # semi-major axis (m)
_WGS84_E2 = 6.6943799901414e-3  # first eccentricity squared


def _lla_to_ecef(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    """LLA → ECEF (m)."""
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z])


def _enu_to_ecef(enu: np.ndarray, lat_rad: float, lon_rad: float) -> np.ndarray:
    """ENU offset → ECEF rotation (no translation)."""
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    # Rotation matrix R: columns are East, North, Up in ECEF
    R = np.array([
        [-sin_lon,          cos_lon,          0.0      ],
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat],
    ])
    return R @ enu


def _ecef_to_lla(ecef: np.ndarray):
    """ECEF → (lat_rad, lon_rad, alt_m) via Bowring iteration."""
    x, y, z = ecef
    lon_rad = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat_rad = math.atan2(z, p * (1.0 - _WGS84_E2))  # initial
    for _ in range(5):
        sin_lat = math.sin(lat_rad)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        lat_rad = math.atan2(z + _WGS84_E2 * N * sin_lat, p)
    sin_lat = math.sin(lat_rad)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    alt_m = p / math.cos(lat_rad) - N if abs(math.cos(lat_rad)) > 1e-10 else abs(z) / sin_lat - N * (1.0 - _WGS84_E2)
    return lat_rad, lon_rad, alt_m


def enu_to_lla(
    east: float, north: float, up: float,
    ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float,
):
    """Convert ENU offset (m) from reference origin to LLA (deg, deg, m)."""
    ref_lat = math.radians(ref_lat_deg)
    ref_lon = math.radians(ref_lon_deg)
    ecef_ref = _lla_to_ecef(ref_lat, ref_lon, ref_alt_m)
    delta_ecef = _enu_to_ecef(np.array([east, north, up]), ref_lat, ref_lon)
    ecef_pt = ecef_ref + delta_ecef
    lat_r, lon_r, alt = _ecef_to_lla(ecef_pt)
    return math.degrees(lat_r), math.degrees(lon_r), alt


# ── Node ──────────────────────────────────────────────────────────────────────


class GpsSimulatorNode(Node):

    def __init__(self) -> None:
        super().__init__("gps_simulator_node")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("update_rate_hz",    5.0)
        self.declare_parameter("ref_lat_deg",       47.3977419)
        self.declare_parameter("ref_lon_deg",       8.5455938)
        self.declare_parameter("ref_alt_m",         486.0)
        self.declare_parameter("noise_h_m",         1.5)
        self.declare_parameter("noise_v_m",         3.0)
        self.declare_parameter("bias_time_const_s", 60.0)
        self.declare_parameter("bias_walk_h_m_s",   0.1)
        self.declare_parameter("bias_walk_v_m_s",   0.15)
        self.declare_parameter("multipath_prob",    0.02)
        self.declare_parameter("multipath_scale",   4.0)
        self.declare_parameter("outage_prob_per_s", 0.0)
        self.declare_parameter("outage_duration_s", 3.0)
        self.declare_parameter("seed",              -1)

        p = self._p  # shorthand

        seed = p("seed", int)
        self._rng = np.random.default_rng(None if seed < 0 else seed)

        # ── State ────────────────────────────────────────────────────────────
        self._bias = np.zeros(3)       # [east, north, up] bias (m)
        self._outage_remaining = 0.0   # seconds of active outage
        self._latest_pose: PoseStamped | None = None

        # ── QoS ──────────────────────────────────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscriber ───────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, "/gt_pub/pose", self._pose_cb, qos_be
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self._fix_pub = self.create_publisher(NavSatFix, "/gps/fix", qos_rel)
        self._enu_pub = self.create_publisher(PointStamped, "/gps/enu", qos_rel)

        # ── Timer at GPS update rate ─────────────────────────────────────────
        rate = p("update_rate_hz", float)
        if rate <= 0.0:
            raise ValueError("update_rate_hz must be positive")
        self._dt = 1.0 / rate
        self.create_timer(self._dt, self._publish_gps)

        self.get_logger().info(
            f"GpsSimulatorNode ready — {rate:.1f} Hz | "
            f"σ_h={p('noise_h_m', float):.2f} m  σ_v={p('noise_v_m', float):.2f} m | "
            f"ref=({p('ref_lat_deg', float):.6f}°, {p('ref_lon_deg', float):.6f}°, "
            f"{p('ref_alt_m', float):.1f} m)"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _p(self, name: str, cast=None):
        val = self.get_parameter(name).value
        return cast(val) if cast else val

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._latest_pose = msg

    def _step_bias(self) -> None:
        """Propagate Gauss-Markov bias one GPS time step."""
        tau   = self._p("bias_time_const_s", float)
        q_h   = self._p("bias_walk_h_m_s", float)
        q_v   = self._p("bias_walk_v_m_s", float)
        dt    = self._dt
        alpha = math.exp(-dt / tau) if tau > 0 else 0.0
        # Process noise std for discrete step: σ_process = q * sqrt(dt)
        noise_h = q_h * math.sqrt(dt)
        noise_v = q_v * math.sqrt(dt)
        drive = self._rng.normal(
            0.0, [noise_h, noise_h, noise_v]
        )
        self._bias = alpha * self._bias + drive

    def _publish_gps(self) -> None:
        if self._latest_pose is None:
            return

        p = self._p
        dt = self._dt

        # ── Outage logic ─────────────────────────────────────────────────────
        outage_prob = p("outage_prob_per_s", float)
        if self._outage_remaining > 0.0:
            self._outage_remaining -= dt
            self._publish_no_fix(self._latest_pose.header.stamp)
            return
        if outage_prob > 0.0:
            # Probability per sample = 1 - exp(-λ·dt)  (Poisson process)
            if self._rng.random() < 1.0 - math.exp(-outage_prob * dt):
                self._outage_remaining = p("outage_duration_s", float)
                self.get_logger().info(
                    f"GPS outage started ({self._outage_remaining:.1f} s)"
                )
                self._publish_no_fix(self._latest_pose.header.stamp)
                return

        # ── Get true ENU position ─────────────────────────────────────────────
        pos = self._latest_pose.pose.position
        enu_true = np.array([pos.x, pos.y, pos.z])

        # ── Step bias ────────────────────────────────────────────────────────
        self._step_bias()

        # ── White noise ──────────────────────────────────────────────────────
        sigma_h = p("noise_h_m", float)
        sigma_v = p("noise_v_m", float)
        white = self._rng.normal(0.0, [sigma_h, sigma_h, sigma_v])

        # ── Multipath ────────────────────────────────────────────────────────
        multipath = np.zeros(3)
        if self._rng.random() < p("multipath_prob", float):
            scale = p("multipath_scale", float) * sigma_h
            multipath = self._rng.normal(0.0, [scale, scale, scale * 0.5])
            self.get_logger().debug(
                f"Multipath event: {np.linalg.norm(multipath[:2]):.2f} m"
            )

        # ── Corrupted ENU ────────────────────────────────────────────────────
        enu_gps = enu_true + self._bias + white + multipath

        # ── ENU → LLA ────────────────────────────────────────────────────────
        lat, lon, alt = enu_to_lla(
            enu_gps[0], enu_gps[1], enu_gps[2],
            p("ref_lat_deg", float),
            p("ref_lon_deg", float),
            p("ref_alt_m", float),
        )

        stamp = self._latest_pose.header.stamp

        # ── NavSatFix ────────────────────────────────────────────────────────
        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = "gps"
        fix.status.status  = NavSatStatus.STATUS_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude  = lat
        fix.longitude = lon
        fix.altitude  = alt

        # Covariance in ENU order (east, north, up) → (lat, lon, alt)
        # Approximate: σ_lat ≈ σ_north / R_earth, σ_lon ≈ σ_east / (R_earth · cos(lat))
        R_earth = _WGS84_A
        cos_lat = math.cos(math.radians(lat))
        var_lat = (sigma_v / R_earth) ** 2   # reuse v for alt diagonal
        # Full covariance diagonal in (lat_rad, lon_rad, alt_m)
        # NavSatFix uses (lat, lon, alt) in metres-equivalent row/col
        # Convention: covariance_type = COVARIANCE_TYPE_DIAGONAL_KNOWN
        cov_lat = (sigma_h / R_earth) ** 2
        cov_lon = (sigma_h / (R_earth * cos_lat + 1e-9)) ** 2
        cov_alt = sigma_v ** 2
        fix.position_covariance = [
            cov_lat, 0.0, 0.0,
            0.0, cov_lon, 0.0,
            0.0, 0.0, cov_alt,
        ]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self._fix_pub.publish(fix)

        # ── Debug ENU point ──────────────────────────────────────────────────
        pt = PointStamped()
        pt.header.stamp = stamp
        pt.header.frame_id = "map"
        pt.point.x = float(enu_gps[0])
        pt.point.y = float(enu_gps[1])
        pt.point.z = float(enu_gps[2])
        self._enu_pub.publish(pt)

    def _publish_no_fix(self, stamp) -> None:
        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = "gps"
        fix.status.status  = NavSatStatus.STATUS_NO_FIX
        fix.status.service = NavSatStatus.SERVICE_GPS
        fix.latitude = fix.longitude = fix.altitude = float("nan")
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self._fix_pub.publish(fix)


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = GpsSimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
