"""
Launch file for feature tracking visualization only.

Usage:
    # Terminal A: play bag
    ros2 bag play bags/euroc_mav0 --rate 0.5 --clock

    # Terminal B: launch feature tracker
    ros2 launch vio_pipeline features.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('vio_pipeline')
    default_config = os.path.join(pkg_share, 'config', 'euroc_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to euroc_params.yaml'),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time from bag'),

        Node(
            package='vio_pipeline',
            executable='feature_tracking_node',
            name='feature_tracking_node',
            output='screen',
            parameters=[{
                'config_path': LaunchConfiguration('config_file'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
        ),
    ])
