"""
Full VIO pipeline launch with RViz visualization.

Works with both EuRoC and UZH bags — pass the matching config file:

    # EuRoC (default)
    ros2 bag play bags/euroc_mav0 --rate 0.5 --clock
    ros2 launch vio_pipeline full_pipeline.launch.py

    # UZH indoor
    ros2 bag play bags/uzh_indoor_9 --rate 0.5 --clock
    ros2 launch vio_pipeline full_pipeline.launch.py \\
        config_file:=/path/to/uzh_indoor_params.yaml
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# GPS profiles are loaded from config/gps_profiles.yaml at launch time.
# Edit that file to change receiver characteristics or add new profiles.


def _make_nodes(context, *args, **kwargs):
    config_file  = LaunchConfiguration("config_file").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    use_tf       = LaunchConfiguration("use_tf_publisher").perform(context)
    use_gps      = LaunchConfiguration("use_gps").perform(context)
    gps_mode     = LaunchConfiguration("gps_mode").perform(context).lower()
    use_sim_time_bool = use_sim_time.lower() in ("true", "1", "yes")
    use_tf_bool       = use_tf.lower()  in ("true", "1", "yes")
    use_gps_bool      = use_gps.lower() in ("true", "1", "yes")

    # Load GPS profiles from YAML
    pkg_share = get_package_share_directory("vio_pipeline")
    gps_profiles_path = os.path.join(pkg_share, "config", "gps_profiles.yaml")
    with open(gps_profiles_path) as f:
        all_profiles = yaml.safe_load(f)
    if gps_mode not in all_profiles:
        raise ValueError(
            f"Unknown gps_mode '{gps_mode}'. "
            f"Valid options: {list(all_profiles)}  (defined in gps_profiles.yaml)"
        )
    gps_sim  = all_profiles[gps_mode]["simulator"]
    gps_eskf = all_profiles[gps_mode]["eskf"]

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    imu_rate_hz = int(cfg.get("imu", {}).get("rate_hz", 200))

    output_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "tmp"
        )
    )

    return [
        # Pose Estimation Node
        Node(
            package="vio_pipeline",
            executable="pose_estimation_node",
            name="pose_estimation_node",
            output="screen",
            parameters=[
                {
                    "config_path": config_file,
                    "use_sim_time": use_sim_time_bool,
                }
            ],
        ),
        # IMU Processing Node
        Node(
            package="vio_pipeline",
            executable="imu_processing_node",
            name="imu_processing_node",
            output="screen",
            parameters=[
                {
                    "config_path": config_file,
                    "use_sim_time": use_sim_time_bool,
                    "imu_topic": "/imu0",
                    "init_duration": .0,
                    "gyro_lpf_cutoff": 15.0,
                    "accel_lpf_cutoff": 10.0,
                    "imu_rate_hz": imu_rate_hz,
                }
            ],
        ),
        # Ground Truth Publisher
        Node(
            package="vio_pipeline",
            executable="ground_truth_publisher",
            name="ground_truth_publisher",
            output="screen",
            parameters=[
                {
                    "config_path": config_file,
                    "use_sim_time": use_sim_time_bool,
                }
            ],
        ),
        # ESKF Node
        # ── Measurement noise ─────────────────────────────────────────────────
        # meas_pos_std  [m]    1-σ noise on VIO position.
        #   Lower  → filter trusts VIO position more, tracks it closely.
        #   Higher → filter relies on IMU integration; smooth but drifts faster.
        #   Typical range: 0.02 (good VIO) … 0.5 (noisy/sparse VIO).
        #
        # meas_ang_std  [rad]  1-σ noise on VIO orientation (≈ degrees × π/180).
        #   Lower  → filter trusts VIO rotation more.
        #   Higher → IMU gyro integration dominates attitude; fine for short runs.
        #   Typical range: 0.01 (≈0.6°) … 0.1 (≈5.7°).
        #
        # ── Initial covariance (P₀ diagonal) ─────────────────────────────────
        # These seed the filter's uncertainty at t=0.  They only affect the
        # first few VIO updates; after ~5 updates the filter self-calibrates.
        #
        # init_pos_std  [m]    Initial position uncertainty.
        #   Larger → first VIO update pulls position correction strongly.
        #   Set ≥ expected VO displacement on the first frame.
        #
        # init_vel_std  [m/s]  Initial velocity uncertainty.
        #   Larger → filter allows large initial velocity estimates.
        #   Drone starts stationary → 0.1–0.5 m/s is generous but safe.
        #
        # init_att_std  [rad]  Initial attitude uncertainty.
        #   Larger → first VIO rotation update has stronger influence.
        #   Keep ≥ gravity-alignment error from static init (≈ 0.02–0.1 rad).
        #
        # init_ba_std   [m/s²] Initial accel-bias uncertainty.
        #   Larger → filter learns bias faster but noisier velocity on startup.
        #   Should bracket expected residual bias after imu_processing removes
        #   the static mean (EuRoC residual ≈ 0.005–0.02 m/s²).
        #
        # init_bg_std   [rad/s] Initial gyro-bias uncertainty.
        #   Larger → filter learns gyro drift faster at the cost of early noise.
        #   EuRoC residual gyro bias ≈ 1e-4–5e-4 rad/s after static removal.
        #
        # ── Numerical guard ───────────────────────────────────────────────────
        # max_dt  [s]  Discard IMU samples whose dt exceeds this threshold.
        #   Prevents covariance blow-up after bag gaps or node restarts.
        #   Should be ≥ 2× nominal IMU period (5 ms → 0.01 s minimum).
        Node(
            package="vio_pipeline",
            executable="eskf_node",
            name="eskf_node",
            output="screen",
            parameters=[
                {
                    "config_path": config_file,
                    "use_sim_time": use_sim_time_bool,
                    # Measurement noise
                    # meas_pos_std: how much to trust each VIO position measurement.
                    #   EuRoC stereo VIO position accuracy ≈ 2–5 cm RMS → 0.05 m.
                    #   With standalone GPS at 1.5 m std, filter trusts VIO 30× more.
                    #   Increase (e.g. 0.15 m) if VIO is noisy or trajectory is long.
                    "meas_pos_std": 0.1,  # m    — stereo VIO ≈ 2–5 cm RMS on EuRoC
                    # meas_ang_std: how much to trust each VIO rotation measurement.
                    #   Stereo VIO rotation accuracy ≈ 0.5–2° RMS → ~0.03 rad (1.7°).
                    #   0.2 rad (11.5°) was far too pessimistic; gyro dominates unfairly.
                    "meas_ang_std": 0.2,  # rad  — stereo VIO ≈ 0.5–2° RMS on EuRoC
                    # Initial covariance
                    "init_pos_std": 1.0,    # m    — generous; corrected on first update
                    "init_vel_std": 0.3,    # m/s  — drone starts stationary; tighter than before
                    "init_att_std": 0.05,   # rad  — ~2.9°; gravity-align from static buffer
                    "init_ba_std":  0.02,   # m/s² — residual after static bias removal
                    "init_bg_std":  5e-4,   # rad/s — residual gyro drift
                    # Numerical guard
                    "max_dt": 0.1,          # s    — skip IMU steps > 100 ms
                    # GPS fusion — profile set by gps_mode launch argument
                    "use_gps":       True,
                    "gps_pos_std_h": gps_eskf["pos_std_h"],
                    "gps_pos_std_v": gps_eskf["pos_std_v"],
                    "gps_gate_chi2": gps_eskf["gate_chi2"],
                }
            ],
        ),
        # TF Publisher (use_tf_publisher:=true to enable)
        *([Node(
            package="vio_pipeline",
            executable="tf_publisher_node",
            name="tf_publisher_node",
            output="screen",
            parameters=[
                {
                    "config_path": config_file,
                    "use_sim_time": use_sim_time_bool,
                }
            ],
        )] if use_tf_bool else []),
        # GPS Simulator (use_gps:=true to enable)
        *([Node(
            package="vio_pipeline",
            executable="gps_simulator_node",
            name="gps_simulator_node",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time_bool,
                    # All params loaded from config/gps_profiles.yaml
                    "update_rate_hz":    gps_sim["update_rate_hz"],
                    "ref_lat_deg":       gps_sim["ref_lat_deg"],
                    "ref_lon_deg":       gps_sim["ref_lon_deg"],
                    "ref_alt_m":         gps_sim["ref_alt_m"],
                    "noise_h_m":         gps_sim["noise_h_m"],
                    "noise_v_m":         gps_sim["noise_v_m"],
                    "bias_time_const_s": gps_sim["bias_time_const_s"],
                    "bias_walk_h_m_s":   gps_sim["bias_walk_h_m_s"],
                    "bias_walk_v_m_s":   gps_sim["bias_walk_v_m_s"],
                    "multipath_prob":    gps_sim["multipath_prob"],
                    "multipath_scale":   gps_sim["multipath_scale"],
                    "outage_prob_per_s": gps_sim["outage_prob_per_s"],
                    "outage_duration_s": gps_sim["outage_duration_s"],
                    # RNG seed (-1 = random)
                    "seed":              -1,
                }
            ],
        )] if use_gps_bool else []),
        # Debug Logger
        Node(
            package="vio_pipeline",
            executable="debug_logger_node",
            name="debug_logger_node",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time_bool,
                    "output_dir": output_dir,
                }
            ],
        ),
        # RViz
        # Node(
        #    package='rviz2',
        #    executable='rviz2',
        #    name='rviz2',
        #    output='screen',
        #    parameters=[{
        #        'use_sim_time': use_sim_time_bool,
        #    }],
        # ),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory("vio_pipeline")
    default_config = os.path.join(pkg_share, "config", "euroc_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to dataset params yaml (euroc_params.yaml or uzh_indoor_params.yaml)",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation time from bag",
            ),
            DeclareLaunchArgument(
                "use_tf_publisher",
                default_value="false",
                description="Launch tf_publisher_node (map→base_link + sensor static TFs)",
            ),
            DeclareLaunchArgument(
                "use_gps",
                default_value="false",
                description="Launch gps_simulator_node (samples /gt_pub/pose, publishes /gps/fix)",
            ),
            DeclareLaunchArgument(
                "gps_mode",
                default_value="standalone",
                description=(
                    "GPS accuracy profile: 'standalone' (1.2 m H / 1.9 m V RMS) "
                    "or 'rtk' (~1 cm H / ~2 cm V, baseline <40 km). "
                    "Sets noise params for both the simulator and the ESKF."
                ),
            ),
            OpaqueFunction(function=_make_nodes),
        ]
    )
