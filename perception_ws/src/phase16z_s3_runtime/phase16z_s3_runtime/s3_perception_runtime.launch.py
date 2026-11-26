#!/usr/bin/env python3
"""Phase16Z 专用 perception_user 启动文件。

只在 Phase16Z 诊断中使用，不修改 Phase16Y 或官方仓库。随机场景仍由同一
WSL/容器内的 VRX 生成器产生；启动前在生成的 SDF 中补齐 GPS/IMU 世界系统，
并在生成模型后再创建 payload bridge，确保 canonical GPS/IMU 话题真实可用。
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

import vrx_gz.launch
from vrx_gz.model import Model


ACTUAL_WORLD_NAME = "perception_user"
SENSOR_PLUGIN_LINE = (
    '    <plugin filename="gz-sim-sensors-system" '
    'name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>'
)
IMU_PLUGIN_LINE = '    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>'
NAVSAT_PLUGIN_LINE = (
    '    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>'
)


def _generate_world(pkg_share: str) -> tuple[Path, str, list[float], int]:
    """在当前 WSL/容器内生成并封存本次运行使用的 SDF。"""
    pkg_prefix = os.path.dirname(os.path.dirname(pkg_share))
    generator = Path(pkg_prefix) / "lib" / "vrx_gz" / "random_world_generator_perception.py"
    if not generator.exists():
        raise RuntimeError(f"PERCEPTION_GENERATOR_NOT_FOUND:{generator}")

    seed_text = os.environ.get("S3_SCENE_SEED", "").strip()
    seed = int(seed_text) if seed_text else random.SystemRandom().randint(0, 99999)
    launch_pid = os.getpid()
    source_path = Path(f"/tmp/phase16z_perception_user_random_{launch_pid}.sdf")
    result = subprocess.run(
        ["python3", str(generator), str(seed), str(source_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"PERCEPTION_GENERATOR_FAILED:{result.returncode}:{result.stderr[-1000:]}"
        )
    if not source_path.exists():
        raise RuntimeError(f"GENERATED_SDF_NOT_FOUND:{source_path}")

    content = source_path.read_text(encoding="utf-8")
    if content.count(SENSOR_PLUGIN_LINE) != 1:
        raise RuntimeError("GENERATED_SDF_SENSOR_PLUGIN_ANCHOR_INVALID")
    patched = content.replace(
        SENSOR_PLUGIN_LINE,
        "\n".join((SENSOR_PLUGIN_LINE, IMU_PLUGIN_LINE, NAVSAT_PLUGIN_LINE)),
        1,
    )
    world_stem = f"phase16z_perception_user_{launch_pid}"
    world_path = Path(pkg_share) / "worlds" / f"{world_stem}.sdf"
    world_path.parent.mkdir(parents=True, exist_ok=True)
    world_path.write_text(patched, encoding="utf-8", newline="\n")

    print(f"PHASE16Z_GENERATOR={generator}", flush=True)
    print(f"PHASE16Z_SCENE_SEED={seed}", flush=True)
    print(f"PHASE16Z_SOURCE_SDF={source_path}", flush=True)
    print(f"PHASE16Z_WORLD_FILE={world_path}", flush=True)
    print("PHASE16Z_WORLD_NAME=perception_user", flush=True)
    print("PHASE16Z_INJECTED_WORLD_PLUGINS=Imu,NavSat", flush=True)
    for line in result.stdout.splitlines():
        print(line, flush=True)
    return world_path, world_stem, _parse_wamv_pose(result.stdout), seed


def _parse_wamv_pose(output: str) -> list[float]:
    for line in output.splitlines():
        if line.startswith("WAMV="):
            values = [float(value) for value in line[5:].split(",")]
            if len(values) == 6:
                return values
    return [-710.0, 350.0, 0.0, 0.0, 0.0, 0.0]


def launch(context, *args, **kwargs):
    world_name = LaunchConfiguration("world").perform(context)
    if os.path.splitext(os.path.basename(world_name))[0] != ACTUAL_WORLD_NAME:
        raise RuntimeError(f"PHASE16Z_WORLD_MUST_BE_{ACTUAL_WORLD_NAME}")

    sim_mode = LaunchConfiguration("sim_mode").perform(context)
    bridge_topics = LaunchConfiguration("bridge_competition_topics").perform(context).lower() == "true"
    competition_mode = LaunchConfiguration("competition_mode").perform(context).lower() == "true"
    robot = LaunchConfiguration("robot").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower() == "true"
    paused = LaunchConfiguration("paused").perform(context).lower() == "true"
    extra_gz_args = LaunchConfiguration("extra_gz_args").perform(context)
    urdf = LaunchConfiguration("urdf").perform(context).strip()

    pkg_share = get_package_share_directory("vrx_gz")
    world_path, world_stem, wamv_pose, seed = _generate_world(pkg_share)
    if not urdf:
        urdf = str(Path(get_package_share_directory("wamv_gazebo")) / "urdf" / "wamv_gazebo.urdf.xacro")

    model = Model("wamv", "wam-v", wamv_pose)
    model.set_urdf(urdf)
    # competition.launch.py 原先在创建 bridge 后才解析 URDF，导致 payload 为空。
    # 先生成一次，使五路 LiDAR、GPS、IMU 都按 canonical ROS 话题创建 bridge。
    model.generate()
    print(f"PHASE16Z_MODEL_URDF={urdf}", flush=True)
    print(f"PHASE16Z_MODEL_PAYLOAD_COUNT={len(model.payload)}", flush=True)
    print(f"PHASE16Z_MODEL_PAYLOAD_NAMES={sorted(model.payload)}", flush=True)
    print(f"PHASE16Z_WORLD_SEED_CONFIRMED={seed}", flush=True)

    launch_processes = []
    launch_processes.extend(vrx_gz.launch.simulation(world_stem, headless, paused, extra_gz_args))
    launch_processes.extend(vrx_gz.launch.spawn(sim_mode, ACTUAL_WORLD_NAME, [model], robot))
    if sim_mode in ("bridge", "full") and bridge_topics:
        launch_processes.extend(vrx_gz.launch.competition_bridges(ACTUAL_WORLD_NAME, competition_mode))
    return launch_processes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=ACTUAL_WORLD_NAME),
            DeclareLaunchArgument("sim_mode", default_value="full"),
            DeclareLaunchArgument("bridge_competition_topics", default_value="True"),
            DeclareLaunchArgument("robot", default_value=""),
            DeclareLaunchArgument("headless", default_value="True"),
            DeclareLaunchArgument("urdf", default_value=""),
            DeclareLaunchArgument("paused", default_value="False"),
            DeclareLaunchArgument("competition_mode", default_value="False"),
            DeclareLaunchArgument("extra_gz_args", default_value=""),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            OpaqueFunction(function=launch),
        ]
    )
