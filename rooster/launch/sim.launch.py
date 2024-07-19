from pathlib import Path

from ament_index_python.packages import get_package_share_directory as get_share_dir
from launch import LaunchDescription
from launch.actions import EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    visuals = IncludeLaunchDescription(
        str(
            Path(get_share_dir("rooster_viz"))
            .joinpath("launch")
            .joinpath("rooster_viz.launch.py")
        )
    )
    rosbag = IncludeLaunchDescription(
        str(
            Path(get_share_dir("rooster"))
            .joinpath("launch")
            .joinpath("rosbag.launch.py")
        )
    )
    simulator_node = Node(
        package="rooster",
        executable="simulator",
        name="simulator",
        output="screen",
        parameters=[
            {
                "ux_mps": 2.5,
                "uy_mps": 0.0,
                "r_radps": 0.0,
                "dfz_long_kn": 0.0,
                "dfz_lat_kn": 0.0,
                "delta_rad": 0.0,
                "fx_kn": 0.0,
                "sim_time_s": 600.0,
                "enable_mpc_bool": 1.0,
                "east_m": -292.2,
                "north_m": -408.2,
                "psi_rad": -3.1,
            }
        ],
        remappings=[
            ("/simulator/to_autobox", "/auto_bridge2/to_autobox"),
            ("/simulator/from_autobox", "/auto_bridge2/from_autobox"),
        ],
    )


    return LaunchDescription(
        [
            RegisterEventHandler(
                OnProcessExit(
                    target_action=simulator_node,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason="Reached end of simulation")),
                    ],
                )
            ),
            Node(
                package="rooster",
                executable="mpc_runner",
                name="mpc_runner",
                output="screen",
            ),
            Node(
                package="rooster",
                executable="coordinator",
                name="coordinator",
                output="screen",
            ),
            rosbag,
            simulator_node,
        ]
    )
