"""启动五路点云靠泊感知。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("phase17_berthing_perception"))
    default_config = str(share / "config" / "berthing_perception.yaml")
    default_environment_config = str(share / "config" / "local_environment.yaml")
    default_extrinsics = str(share / "config" / "static_extrinsics.yaml")
    default_template = str(share / "config" / "full_dock_collision_points.csv")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("environment_config", default_value=default_environment_config),
            DeclareLaunchArgument("receipt_path", default_value=""),
            DeclareLaunchArgument("environment_receipt_path", default_value=""),
            Node(
                package="phase17_berthing_perception",
                executable="berthing_perception_node",
                name="phase17_berthing_perception",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "extrinsics_path": default_extrinsics,
                        "dock_template_path": default_template,
                        "receipt_path": LaunchConfiguration("receipt_path"),
                    },
                ],
            ),
            Node(
                package="phase17_berthing_perception",
                executable="local_environment_node",
                name="phase17_local_environment",
                output="screen",
                parameters=[
                    LaunchConfiguration("environment_config"),
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "extrinsics_path": default_extrinsics,
                        "footprint_path": str(share / "config" / "hull_collision_footprint_base_link.yaml"),
                        "receipt_path": LaunchConfiguration("environment_receipt_path"),
                    },
                ],
            ),
        ]
    )
