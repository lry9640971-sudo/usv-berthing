"""复用官方仿真、船模和桥接；文件名与内部 world 名分开传递。"""

import json
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from vrx_gz import launch as official
from vrx_gz.model import Model


def setup(context):
    manifest = json.loads(Path(LaunchConfiguration("manifest").perform(context)).read_text(encoding="utf-8"))
    model = Model("wamv", "wam-v", manifest["wamv_pose"])
    # 传感器、物理及障碍运动均来自官方原始配置，不改频率、线数或场景难度。
    rendering = LaunchConfiguration("headless_rendering").perform(context) == "true"
    actions = official.simulation(str(Path(manifest["world_path"]).with_suffix("")), headless=True, paused=True,
                                 extra_gz_args="--headless-rendering" if rendering else "")
    actions += official.spawn("full", manifest["world_name"], [model], "")
    actions += official.competition_bridges(manifest["world_name"], False)
    tracker = Path(get_package_prefix("vrx_gz")) / "lib/vrx_gz/obstacle_tracker.py"
    dock = manifest["dock_pose"]
    actions.append(ExecuteProcess(cmd=["python3", "-u", str(tracker), "--ros-args",
                                       "-p", "world:=" + manifest["world_name"],
                                       "-p", "world_sdf:=" + manifest["world_path"],
                                       "-p", f"dock_x:={dock[0]}", "-p", f"dock_y:={dock[1]}",
                                       "-p", f"dock_yaw:={dock[5]}"], output="screen"))
    return actions


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("manifest"),
                              DeclareLaunchArgument("headless_rendering", default_value="false"),
                              OpaqueFunction(function=setup)])
