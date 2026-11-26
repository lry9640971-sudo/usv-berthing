"""检查 D0 七雷达到六相机外参链及其退化状态。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


CAMERAS = ("front", "back", "left", "left2", "right", "right2")


def rotation_error(matrix: np.ndarray) -> float:
    value = (float(np.trace(matrix)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def check_contract(candidate_path: str | Path, camera_path: str | Path) -> dict:
    candidate = yaml.safe_load(Path(candidate_path).read_text(encoding="utf-8"))
    camera = yaml.safe_load(Path(camera_path).read_text(encoding="utf-8"))
    lidar = candidate.get("candidate_extrinsics", {})
    transforms = camera.get("transforms", camera.get("camera_extrinsics", {}))
    rows = []
    for name in CAMERAS:
        lidar_name = f"{name}_lidar_sensor"
        value = transforms.get(name)
        available = lidar_name in lidar and value is not None
        finite = False
        orthogonality = None
        determinant = None
        if available:
            matrix = np.asarray(value, dtype=float)
            finite = bool(matrix.shape == (4, 4) and np.isfinite(matrix).all())
            if finite:
                r = matrix[:3, :3]
                orthogonality = float(np.max(np.abs(r.T @ r - np.eye(3))))
                determinant = float(np.linalg.det(r))
        rows.append(
            {
                "camera": name,
                "paired_lidar": lidar_name,
                "available": available,
                "finite": finite,
                "orthogonality_error": orthogonality,
                "determinant": determinant,
                "pass": bool(
                    available
                    and finite
                    and orthogonality is not None
                    and orthogonality <= 1e-6
                    and determinant is not None
                    and abs(determinant - 1.0) <= 1e-6
                ),
            }
        )
    all_pass = all(row["pass"] for row in rows)
    return {
        "schema_version": 1,
        "state": "CALIBRATION_CONTRACT_VALID" if all_pass else "CALIBRATION_DEGRADED",
        "candidate_path": str(Path(candidate_path)),
        "camera_path": str(Path(camera_path)),
        "camera_count": len(rows),
        "valid_count": sum(row["pass"] for row in rows),
        "rows": rows,
        "note": "该检查验证外参链结构；自动恢复精度由 D0 独立开发/验证报告给出。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查七雷达到六相机外参链")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--camera-extrinsics", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = check_contract(args.candidate, args.camera_extrinsics)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if report["state"] == "CALIBRATION_CONTRACT_VALID" else 2)


if __name__ == "__main__":
    main()
