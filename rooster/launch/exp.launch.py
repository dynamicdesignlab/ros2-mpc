from pathlib import Path

from ament_index_python.packages import get_package_share_directory as get_share_dir
from launch_ros.actions import Node

import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription


def generate_launch_description():
    rosbag = IncludeLaunchDescription(
        str(
            Path(get_share_dir("rooster"))
            .joinpath("launch")
            .joinpath("rosbag.launch.py")
        )
    )

    return LaunchDescription(
        [
            rosbag,
            Node(
                package="rooster",
                executable="mpc_runner",
                name="mpc_runner",
            ),
            Node(
                package="rooster",
                executable="coordinator",
                name="coordinator",
            ),
            Node(
                package="auto_bridge2",
                executable="auto_bridge2",
                name="auto_bridge2",
            ),
        ]
    )
