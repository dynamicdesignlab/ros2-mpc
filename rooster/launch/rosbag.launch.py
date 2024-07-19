import launch
from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            launch.actions.ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "record",
                    "/auto_bridge2/from_autobox",
                    "/auto_bridge2/to_autobox",
                    "/rooster/nlp_out",
                ]
            ),
        ]
    )
