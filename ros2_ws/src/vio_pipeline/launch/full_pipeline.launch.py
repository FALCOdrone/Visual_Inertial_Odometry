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

Tunable parameters (ESKF noise, IMU filter cutoffs, KLT settings, VIO thresholds)
live in config/pipeline_params.yaml — edit that file to change behaviour.
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
    config_file          = LaunchConfiguration("config_file").perform(context)
    pipeline_params_file = LaunchConfiguration("pipeline_params_file").perform(context)
    use_sim_time  = LaunchConfiguration("use_sim_time").perform(context)
    use_tf        = LaunchConfiguration("use_tf_publisher").perform(context)
    use_gps       = LaunchConfiguration("use_gps").perform(context)
    gps_mode      = LaunchConfiguration("gps_mode").perform(context).lower()
    use_rectifier = LaunchConfiguration("use_rectifier").perform(context)
    use_fgo       = LaunchConfiguration("use_fgo").perform(context)
    use_sim_time_bool  = use_sim_time.lower()  in ("true", "1", "yes")
    use_tf_bool        = use_tf.lower()        in ("true", "1", "yes")
    use_gps_bool       = use_gps.lower()       in ("true", "1", "yes")
    use_rectifier_bool = use_rectifier.lower() in ("true", "1", "yes")
    use_fgo_bool       = use_fgo.lower()       in ("true", "1", "yes")

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

    # Load tunable pipeline parameters from pipeline_params.yaml
    with open(pipeline_params_file, "r") as f:
        pp = yaml.safe_load(f)
    eskf_p = pp["eskf"]
    imu_p  = pp["imu"]
    ft_p   = pp["feature_tracking"]
    vio_p  = pp["vio"]
    fgo_p  = pp.get("fgo", {})

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
                    # Feature tracking params (passed to FeatureExtractor)
                    "max_corners":       int(ft_p["max_corners"]),
                    "quality_level":     float(ft_p["quality_level"]),
                    "min_distance":      int(ft_p["min_distance"]),
                    "win_size_w":        int(ft_p["win_size"][0]),
                    "win_size_h":        int(ft_p["win_size"][1]),
                    "max_level":         int(ft_p["max_level"]),
                    "max_epipolar_err":  float(ft_p["max_epipolar_err"]),
                    # VIO pose estimation params
                    "min_tracks":               int(vio_p["min_tracks"]),
                    "circular_check_threshold": float(vio_p["circular_check_threshold"]),
                    "max_depth":                float(vio_p["max_depth"]),
                    "min_inlier_ratio":         float(vio_p["min_inlier_ratio"]),
                    "max_translation":          float(vio_p["max_translation"]),
                    "max_rotation_deg":         float(vio_p["max_rotation_deg"]),
                    "kf_min_translation":  float(vio_p["kf_min_translation"]),
                    "kf_min_rotation_deg": float(vio_p["kf_min_rotation_deg"]),
                    "kf_max_frames":       int(vio_p["kf_max_frames"]),
                    "pose_cov_pos_base":   float(vio_p["pose_cov_pos_base"]),
                    "pose_cov_ang_base":   float(vio_p["pose_cov_ang_base"]),
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
                    "imu_topic":        "/imu0",
                    "init_duration":    float(imu_p["init_duration"]),
                    "gyro_lpf_cutoff":  float(imu_p["gyro_lpf_cutoff"]),
                    "accel_lpf_cutoff": float(imu_p["accel_lpf_cutoff"]),
                    "imu_rate_hz":      imu_rate_hz,
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
        # ESKF Node (skipped when use_fgo:=true)
        *([Node(
            package="vio_pipeline",
            executable="eskf_node",
            name="eskf_node",
            output="screen",
            parameters=[
                {
                    "config_path": config_file,
                    "use_sim_time": use_sim_time_bool,
                    "meas_pos_std": float(eskf_p["meas_pos_std"]),
                    "meas_ang_std": float(eskf_p["meas_ang_std"]),
                    "init_pos_std": float(eskf_p["init_pos_std"]),
                    "init_vel_std": float(eskf_p["init_vel_std"]),
                    "init_att_std": float(eskf_p["init_att_std"]),
                    "init_ba_std":  float(eskf_p["init_ba_std"]),
                    "init_bg_std":  float(eskf_p["init_bg_std"]),
                    "max_dt":       float(eskf_p["max_dt"]),
                    "use_gps":       False,
                    "gps_pos_std_h": gps_eskf["pos_std_h"],
                    "gps_pos_std_v": gps_eskf["pos_std_v"],
                    "gps_gate_chi2": gps_eskf["gate_chi2"],
                }
            ],
        )] if not use_fgo_bool else []),
        # FGO Backend Node (use_fgo:=true to enable)
        *([Node(
            package="vio_pipeline",
            executable="fgo_backend_node",
            name="fgo_backend_node",
            output="screen",
            parameters=[
                {
                    "config_path": config_file,
                    "use_sim_time": use_sim_time_bool,
                    "window_size":    int(fgo_p.get("window_size", 10)),
                    "lm_max_iter":    int(fgo_p.get("lm_max_iter", 8)),
                    "lm_lambda_init": float(fgo_p.get("lm_lambda_init", 1e-3)),
                    "imu_noise_scale": float(fgo_p.get("imu_noise_scale", 20.0)),
                    "min_gravity_samples": int(fgo_p.get("min_gravity_samples", 100)),
                    "vo_pos_std":     float(fgo_p.get("vo_pos_std", 0.05)),
                    "vo_ang_std":     float(fgo_p.get("vo_ang_std", 0.05)),
                    "prior_pos_std":  float(fgo_p.get("prior_pos_std", 0.01)),
                    "prior_vel_std":  float(fgo_p.get("prior_vel_std", 1.0)),
                    "prior_att_std":  float(fgo_p.get("prior_att_std", 0.01)),
                    "prior_ba_std":   float(fgo_p.get("prior_ba_std", 0.02)),
                    "prior_bg_std":   float(fgo_p.get("prior_bg_std", 5e-4)),
                }
            ],
        )] if use_fgo_bool else []),
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
        # Stereo Rectifier (use_rectifier:=true to enable)
        *([Node(
            package="vio_pipeline",
            executable="stereo_rectifier_node",
            name="stereo_rectifier_node",
            output="screen",
            parameters=[
                {
                    "config_path": config_file,
                    "use_sim_time": use_sim_time_bool,
                }
            ],
        )] if use_rectifier_bool else []),
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
                    # Path to pipeline_params.yaml so the logger can copy it to tmp/
                    "pipeline_params_file": pipeline_params_file,
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
    default_pipeline_params = os.path.join(pkg_share, "config", "pipeline_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to dataset params yaml (euroc_params.yaml or uzh_indoor_params.yaml)",
            ),
            DeclareLaunchArgument(
                "pipeline_params_file",
                default_value=default_pipeline_params,
                description="Path to tunable pipeline parameters yaml (ESKF noise, IMU cutoffs, KLT settings)",
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
                "use_rectifier",
                default_value="true",
                description="Launch stereo_rectifier_node (publishes /cam0/image_rect, /cam1/image_rect)",
            ),
            DeclareLaunchArgument(
                "use_gps",
                default_value="false",
                description="Launch gps_simulator_node (samples /gt_pub/pose, publishes /gps/fix)",
            ),
            DeclareLaunchArgument(
                "use_fgo",
                default_value="false",
                description="Use FGO sliding-window backend instead of ESKF",
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
