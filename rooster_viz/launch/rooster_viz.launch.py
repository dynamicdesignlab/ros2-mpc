import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="rooster_viz",
                executable="vehicle_visualizer",
                name="veh_viz",
            ),
            Node(
                package="rooster_viz",
                executable="world_visualizer",
                name="world_viz",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=[
                    "-d",
                    os.path.join(
                        get_package_share_directory("rooster_viz"),
                        "config",
                        "rooster_viz.rviz",
                    ),
                ],
            ),
        ]
    )
