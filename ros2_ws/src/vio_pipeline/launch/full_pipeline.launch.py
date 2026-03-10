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

    # With FGO backend
    ros2 launch vio_pipeline full_pipeline.launch.py use_fgo:=true
"""

import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _make_nodes(context, *args, **kwargs):
    config_file = LaunchConfiguration("config_file").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    use_sim_time_bool = use_sim_time.lower() in ("true", "1", "yes")
    use_fgo = LaunchConfiguration("use_fgo").perform(context)

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    imu_rate_hz = int(cfg.get("imu", {}).get("rate_hz", 200))

    output_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "tmp"
        )
    )

    nodes = [
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
                    "init_duration": 2.0,
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
        # TF Publisher
        Node(
            package="vio_pipeline",
            executable="tf_publisher_node",
            name="tf_publisher_node",
            output="screen",
            parameters=[
                {
                    "config_path": config_file,
                    "use_sim_time": use_sim_time_bool,
                    "max_matches": 500,
                }
            ],
        ),
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

    # ── Optional FGO backend ───────────────────────────────────────────────────
    if use_fgo.lower() in ("true", "1", "yes"):
        nodes.append(
            Node(
                package="vio_pipeline",
                executable="fgo_backend_node",
                name="fgo_backend_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time_bool,
                        "window_size": 10,
                        "lm_max_iters": 5,
                        "kf_trans_thresh": 0.05,
                        "kf_rot_thresh_deg": 5.0,
                        "kf_time_thresh": 0.25,
                        "sigma_a": 2.0e-3,
                        "sigma_g": 1.6968e-4,
                        "sigma_ba": 3.0e-3,
                        "sigma_bg": 1.9393e-5,
                        "max_dt": 0.1,
                    }
                ],
            )
        )

    return nodes


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
                "use_fgo",
                default_value="false",
                description="Launch the FGO backend node",
            ),
            OpaqueFunction(function=_make_nodes),
        ]
    )
