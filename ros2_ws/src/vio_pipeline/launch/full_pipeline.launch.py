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
