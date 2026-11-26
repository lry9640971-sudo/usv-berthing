"""启动六鱼眼 10 m 环视鸟瞰。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("phase17_surround_bev"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("config", default_value=str(share / "config" / "surround_bev.yaml")),
            DeclareLaunchArgument("receipt_path", default_value=""),
            DeclareLaunchArgument(
                "lut_path", default_value=str(share / "config" / "masked_fixed_geometry_bev_lut.npz")
            ),
            DeclareLaunchArgument(
                "calibration_state", default_value="D0_RELATIVE_EXTRINSICS_RECOVERED_DEV"
            ),
            Node(
                package="phase17_surround_bev",
                executable="surround_bev_node",
                name="phase17_surround_bev",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "lut_path": LaunchConfiguration("lut_path"),
                        "calibration_state": LaunchConfiguration("calibration_state"),
                        "receipt_path": LaunchConfiguration("receipt_path"),
                    },
                ],
            ),
        ]
    )
