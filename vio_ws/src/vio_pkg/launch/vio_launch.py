import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # --- Launch Arguments ---
    bag_path_arg = DeclareLaunchArgument(
        "bag_path",
        default_value="/mnt/d/GITHUB/VIO/easy1",
        description="Absolute path to the rosbag directory (containing .mcap)",
    )

    bag_rate_arg = DeclareLaunchArgument(
        "bag_rate",
        default_value="1.0",
        description="Playback rate for the rosbag (e.g. 0.5 for half speed)",
    )

    # --- Nodes ---
    frontend_node = Node(
        package="vio_pkg",
        executable="frontend",
        name="vio_frontend",
        output="screen",
    )

    backend_node = Node(
        package="vio_pkg",
        executable="backend",
        name="vio_backend",
        output="screen",
    )

    mapper_node = Node(
        package="vio_pkg",
        executable="mapper",
        name="vio_mapper",
        output="screen",
    )

    loop_closure_node = Node(
        package="vio_pkg",
        executable="loop_closure",
        name="vio_loop_closure",
        output="screen",
    )

    ground_truth_node = Node(
        package="vio_pkg",
        executable="ground_truth_pub",
        name="ground_truth_publisher",
        output="screen",
    )

    comparator_node = Node(
        package="vio_pkg",
        executable="trajectory_comparator",
        name="trajectory_comparator",
        output="screen",
    )

    # --- Rosbag Playback ---
    # Delayed slightly to let nodes start up before data flows
    rosbag_play = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "play",
                    LaunchConfiguration("bag_path"),
                    "--rate",
                    LaunchConfiguration("bag_rate"),
                    "--clock",
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            bag_path_arg,
            bag_rate_arg,
            frontend_node,
            backend_node,
            mapper_node,
            loop_closure_node,
            ground_truth_node,
            comparator_node,
            rosbag_play,
        ]
    )
