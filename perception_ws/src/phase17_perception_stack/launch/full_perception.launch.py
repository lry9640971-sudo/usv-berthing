"""统一启动环视、靠泊环境和动态船速度模块。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def _launch_path(package: str, filename: str) -> str:
    return str(Path(get_package_share_directory(package)) / "launch" / filename)


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")
    receipt_dir = LaunchConfiguration("receipt_dir")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("enable_bev", default_value="true"),
            DeclareLaunchArgument("enable_berthing", default_value="true"),
            DeclareLaunchArgument("enable_dynamic_speed", default_value="true"),
            DeclareLaunchArgument("receipt_dir", default_value="/tmp/phase17_perception_receipts"),
            DeclareLaunchArgument("maximum_dynamic_range_m", default_value="80.0"),
            DeclareLaunchArgument("publish_fixed_sensor_tf", default_value="false"),
            DeclareLaunchArgument("enable_gps_velocity", default_value="true"),
            DeclareLaunchArgument("gps_velocity_gz_topic", default_value="/world/berthing_user/model/wamv/link/wamv/gps_wamv_link/sensor/navsat/navsat"),
            DeclareLaunchArgument("gps_velocity_ros_topic", default_value="/wamv/sensors/gps/gps/velocity"),
            GroupAction(
                scoped=True,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            _launch_path("phase17_surround_bev", "surround_bev.launch.py")
                        ),
                        condition=IfCondition(LaunchConfiguration("enable_bev")),
                        launch_arguments={
                            "use_sim_time": use_sim_time,
                            "receipt_path": PathJoinSubstitution(
                                [receipt_dir, "surround_bev_receipt.json"]
                            ),
                        }.items(),
                    )
                ],
            ),
            GroupAction(
                scoped=True,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            _launch_path("phase17_berthing_perception", "berthing_perception.launch.py")
                        ),
                        condition=IfCondition(LaunchConfiguration("enable_berthing")),
                        launch_arguments={
                            "use_sim_time": use_sim_time,
                            "receipt_path": PathJoinSubstitution([receipt_dir, "berthing_receipt.json"]),
                            "environment_receipt_path": PathJoinSubstitution(
                                [receipt_dir, "local_environment_receipt.json"]
                            ),
                        }.items(),
                    )
                ],
            ),
            GroupAction(
                scoped=True,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            _launch_path("phase16z_s3_runtime", "dynamic_speed.launch.py")
                        ),
                        condition=IfCondition(LaunchConfiguration("enable_dynamic_speed")),
                        launch_arguments={
                            "use_sim_time": use_sim_time,
                            "candidate_receipt_path": PathJoinSubstitution(
                                [receipt_dir, "dynamic_candidate_receipt.json"]
                            ),
                            "localization_receipt_path": PathJoinSubstitution(
                                [receipt_dir, "dynamic_localization_receipt.json"]
                            ),
                            "navigation_receipt_path": PathJoinSubstitution(
                                [receipt_dir, "navigation_receipt.json"]
                            ),
                            "enable_gps_velocity": LaunchConfiguration("enable_gps_velocity"),
                            "gps_velocity_gz_topic": LaunchConfiguration("gps_velocity_gz_topic"),
                            "gps_velocity_ros_topic": LaunchConfiguration("gps_velocity_ros_topic"),
                            "maximum_detection_range_m": LaunchConfiguration(
                                "maximum_dynamic_range_m"
                            ),
                            "publish_fixed_sensor_tf": LaunchConfiguration(
                                "publish_fixed_sensor_tf"
                            ),
                        }.items(),
                    )
                ],
            ),
        ]
    )
