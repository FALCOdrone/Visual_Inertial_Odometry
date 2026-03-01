"""
Full VIO pipeline launch with RViz visualization.

Usage:
    # Terminal A: play bag
    ros2 bag play bags/euroc_mav0 --rate 0.5 --clock

    # Terminal B: launch full pipeline
    ros2 launch vio_pipeline full_pipeline.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("vio_pipeline")
    default_config = os.path.join(pkg_share, "config", "euroc_params.yaml")
    output_dir = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "tmp"
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to euroc_params.yaml",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation time from bag",
            ),
            # Pose Estimation Node
            Node(
                package="vio_pipeline",
                executable="pose_estimation_node",
                name="pose_estimation_node",
                output="screen",
                parameters=[
                    {
                        "config_path": LaunchConfiguration("config_file"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
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
                        "config_path": LaunchConfiguration("config_file"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "imu_topic": "/imu0",
                        "init_duration": 5.0,
                        "gyro_lpf_cutoff": 15.0,
                        "accel_lpf_cutoff": 10.0,
                        "imu_rate_hz": 200,
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
                        "config_path": LaunchConfiguration("config_file"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),
            # ESKF Node
            Node(
                package="vio_pipeline",
                executable="eskf_node",
                name="eskf_node",
                output="screen",
                parameters=[
                    {
                        "config_path": LaunchConfiguration("config_file"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "meas_pos_std": 0.05,
                        "meas_ang_std": 0.02,
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
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
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
            #        'use_sim_time': LaunchConfiguration('use_sim_time'),
            #    }],
            #    # Users can save/load rviz config; default launches with empty config
            # ),
        ]
    )
