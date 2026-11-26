"""原样调用官方 berthing_user 生成器，记录可复现种子及独立评估场景。"""

import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix


def main():
    out, seed = Path(sys.argv[1]), int(sys.argv[2])
    scene = out / "scene"
    scene.mkdir(parents=True, exist_ok=False)
    world = scene / "berthing_user.sdf"
    generator = Path(get_package_prefix("vrx_gz")) / "lib/vrx_gz/random_world_generator_berthing.py"
    result = subprocess.run([sys.executable, str(generator), str(seed), str(world)], capture_output=True,
                            text=True, encoding="utf-8", check=True, timeout=30)
    (scene / "generator.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    fields = {line.split("=", 1)[0]: line.split("=", 1)[1] for line in result.stdout.splitlines() if "=" in line}
    wamv, dock = [float(value) for value in fields["WAMV"].split(",")], [float(value) for value in fields["DOCK"].split(",")]
    document = ET.parse(world).getroot().find("world")
    names = [item.get("name", "") for item in document.findall("model")]
    names += [item.findtext("name", "") for item in document.findall("include")]
    static = [name for name in names if name.startswith(("pillar", "buoy", "block"))]
    dynamic = [name for name in names if name.startswith("boat")]
    manifest = {"official_generator": str(generator), "seed": seed, "world_path": str(world),
                "world_name": document.get("name"), "wamv_pose": wamv, "dock_pose": dock,
                "static_obstacle_names": static, "dynamic_obstacle_names": dynamic,
                "production_truth_input": False}
    (scene / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"seed": seed, "world": document.get("name"), "initial_pose": wamv,
                      "static_count": len(static), "dynamic_count": len(dynamic)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
