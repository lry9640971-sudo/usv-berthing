"""从七路 PointCloud2 bag 自动恢复相对外参并生成六相机 BEV LUT。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rosbag2_py
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

from . import d0_solver_core as d0
from . import r1_calibration_core as r1
from . import r3_precheck_core as r3
from .lut_builder import build_masked_lut


TOPICS = {name: f"/wamv/sensors/lidars/{name}/points" for name in d0.NAMES}


def _inverse(transform: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=float)
    output[:3, :3] = transform[:3, :3].T
    output[:3, 3] = -output[:3, :3] @ transform[:3, 3]
    return output


def _collect_headers(bag: Path) -> dict[str, list[tuple[int, int]]]:
    topic_to_name = {topic: name for name, topic in TOPICS.items()}
    rows = {name: [] for name in d0.NAMES}
    ordinal = {name: 0 for name in d0.NAMES}
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topic_to_name)))
    while reader.has_next():
        topic, raw, _record = reader.read_next()
        name = topic_to_name[topic]
        message = deserialize_message(raw, PointCloud2)
        stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        rows[name].append((ordinal[name], stamp))
        ordinal[name] += 1
    return rows


def _make_group(reference, rows, group_id):
    _reference_index, reference_stamp = reference
    topics, stamps = {}, []
    for name in d0.NAMES:
        index, stamp = min(rows[name], key=lambda item: abs(item[1] - reference_stamp))
        if abs(stamp - reference_stamp) > 25_000_000:
            return None
        topics[TOPICS[name]] = {
            "index": int(index),
            "stamp_ns": int(stamp),
            "delta_ms": abs(stamp - reference_stamp) / 1e6,
        }
        stamps.append(stamp)
    return {
        "group_id": group_id,
        "reference_topic": TOPICS["forward_lidar_sensor"],
        "reference_stamp_ns": int(reference_stamp),
        "max_skew_ms": (max(stamps) - min(stamps)) / 1e6,
        "topics": topics,
    }


def build_split_caches(bag: Path, output: Path, groups_per_split: int):
    """从任意标准七雷达 MCAP 自动生成互不重叠的开发/验证时间组。"""
    rows = _collect_headers(bag)
    counts = {name: len(values) for name, values in rows.items()}
    if min(counts.values(), default=0) < groups_per_split * 2:
        raise RuntimeError(f"七路点云帧数不足：{counts}")
    candidates = []
    for index, reference in enumerate(rows["forward_lidar_sensor"]):
        group = _make_group(reference, rows, f"candidate_{index:04d}")
        if group is not None:
            candidates.append(group)
    required = groups_per_split * 2
    if len(candidates) < required:
        raise RuntimeError(f"同步七路点云组不足：{len(candidates)} < {required}")
    indices = np.linspace(0, len(candidates) - 1, required, dtype=int)
    selected = [candidates[int(index)] for index in indices]
    paths = []
    output.mkdir(parents=True, exist_ok=True)
    for split, groups in (("s1", selected[0::2]), ("s2", selected[1::2])):
        for index, group in enumerate(groups):
            group["group_id"] = f"{split}_{index:02d}"
        path = output / f"sampling_{split}.json"
        path.write_text(
            json.dumps({"bag": str(bag), "main_groups": groups}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    diagnostics = {
        "topic_counts": counts,
        "synchronized_candidates": len(candidates),
        "groups_per_split": groups_per_split,
        "max_selected_skew_ms": max(group["max_skew_ms"] for group in selected),
    }
    return paths[0], paths[1], diagnostics


def _rotation_deg(transform: np.ndarray) -> float:
    value = (float(np.trace(transform[:3, :3])) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def _rank_diagnostics(frames, candidate) -> dict:
    blocks = d0.associate_blocks(frames, candidate, d0.EDGES, d0.FINE_RADIUS, d0.FINE_VOXEL)
    variables = [name for name in d0.NAMES if name != d0.GAUGE]
    residual_blocks = d0.residual_jacobian(blocks, variables)
    samples = []
    for _block, _residual, jacobian in residual_blocks:
        if len(jacobian) > 600:
            jacobian = jacobian[np.linspace(0, len(jacobian) - 1, 600, dtype=int)]
        samples.append(jacobian)
    if not samples:
        return {"used_blocks": 0, "rank": 0, "columns": 36, "condition": None}
    matrix = np.vstack(samples)
    scale = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.maximum(scale, 1e-12)[None, :]
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-4)) if singular.size else 0
    return {
        "used_blocks": len(residual_blocks),
        "rank": rank,
        "columns": int(matrix.shape[1]),
        "sample_rows": int(matrix.shape[0]),
        "condition": float(singular[0] / max(singular[-1], 1e-12)),
    }


def _camera_chain(candidate_extrinsics, geometry_path: Path):
    contract = yaml.safe_load(geometry_path.read_text(encoding="utf-8"))
    transforms, rows = {}, []
    for name in ("front", "back", "left", "left2", "right", "right2"):
        camera_lidar = np.asarray(contract["nominal_transforms"][name]["T_camera_lidar"], dtype=float)
        base_lidar = candidate_extrinsics[f"{name}_lidar_sensor"]
        base_camera = base_lidar @ _inverse(camera_lidar)
        transforms[name] = base_camera
        rows.append(
            {
                "camera": name,
                "chain_error": float(np.max(np.abs(base_camera @ camera_lidar - base_lidar))),
            }
        )
    return transforms, rows


def _write_trace(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="七雷达自动外参标定并生成六相机 BEV LUT")
    parser.add_argument("--input-bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-per-split", type=int, default=4)
    parser.add_argument("--source-contract", default="")
    parser.add_argument("--geometry-contract", default="")
    parser.add_argument("--static-mask-lut", default="")
    args = parser.parse_args()
    share = Path(get_package_share_directory("phase17_surround_bev"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source_contract or share / "config" / "source_contract.json")
    geometry_path = Path(args.geometry_contract or share / "config" / "nominal_geometry_contract.yaml")
    mask_path = Path(args.static_mask_lut or share / "config" / "masked_fixed_geometry_bev_lut.npz")
    cache_s1, cache_s2, sampling = build_split_caches(
        Path(args.input_bag), output / "sampling", int(args.groups_per_split)
    )
    sensors = r3.load_sensors(source_path, r1)
    frames_s1 = d0.read_groups(r1, r3, cache_s1, sensors, None, "S1", print)
    frames_s2 = d0.read_groups(r1, r3, cache_s2, sensors, None, "S2", print)
    identity = d0.initial_delta_map()
    levels = [
        ("coarse", (0.50, 0.25 + 24.0 * math.sin(math.radians(2.5)) + 0.10, 0.20)),
        ("mid", (0.30, 0.80, 0.10)),
        ("fine", (0.20, 0.30, 0.05)),
    ]
    trace = []
    candidate, solver = d0.solve_levels(
        frames_s1, identity, d0.EDGES, d0.GAUGE, trace, "auto", levels
    )
    s1_nominal = d0.evaluation_metrics(frames_s1, identity, d0.EDGES)
    s2_nominal = d0.evaluation_metrics(frames_s2, identity, d0.EDGES)
    s1_metric = d0.evaluation_metrics(frames_s1, candidate, d0.EDGES)
    s2_metric = d0.evaluation_metrics(frames_s2, candidate, d0.EDGES)
    selection = "optimized"
    if (
        s2_metric["cost_median"] > 1.05 * s2_nominal["cost_median"]
        or s2_metric["coverage"] < 0.95 * s2_nominal["coverage"]
    ):
        # 独立时间组变差时回退到名义解，同时仍须通过秩检查才允许发布。
        candidate = identity
        selection = "nominal_guard"
        s1_metric = s1_nominal
        s2_metric = s2_nominal
    rank = _rank_diagnostics(frames_s1, candidate)
    corrections = {
        name: {
            "translation_m": float(np.linalg.norm(candidate[name][:3, 3])),
            "rotation_deg": _rotation_deg(candidate[name]),
        }
        for name in d0.NAMES
    }
    max_translation = max(item["translation_m"] for item in corrections.values())
    max_rotation = max(item["rotation_deg"] for item in corrections.values())
    quality_pass = bool(
        rank["rank"] == rank["columns"]
        and rank["used_blocks"] >= 7
        and s2_metric["normal_rows"] >= 1000
        and s2_metric["cost_median"] <= 1.05 * s2_nominal["cost_median"]
        and s2_metric["coverage"] >= 0.95 * s2_nominal["coverage"]
        and max_translation <= 0.75
        and max_rotation <= 15.0
    )
    report = {
        "schema_version": 1,
        "state": "AUTO_CALIBRATION_VALID" if quality_pass else "CALIBRATION_DEGENERATE",
        "input_bag": str(Path(args.input_bag)),
        "gauge": d0.GAUGE,
        "sampling": sampling,
        "solver": solver,
        "selection": selection,
        "rank": rank,
        "s1": {key: value for key, value in s1_metric.items() if key != "rows"},
        "s1_nominal": {key: value for key, value in s1_nominal.items() if key != "rows"},
        "s2": {key: value for key, value in s2_metric.items() if key != "rows"},
        "s2_nominal": {key: value for key, value in s2_nominal.items() if key != "rows"},
        "corrections": corrections,
        "max_translation_m": max_translation,
        "max_rotation_deg": max_rotation,
        "truth_or_exact_tf_consumed": [],
    }
    if quality_pass:
        candidate_extrinsics = {name: candidate[name] @ sensors[name]["T"] for name in d0.NAMES}
        camera_transforms, closure = _camera_chain(candidate_extrinsics, geometry_path)
        candidate_doc = {
            "gauge": d0.GAUGE,
            "transform_convention": "T_candidate_sensor = Delta_sensor @ T_nominal_sensor",
            "delta": {name: candidate[name].tolist() for name in d0.NAMES},
            "nominal_extrinsics": {name: sensors[name]["T"].tolist() for name in d0.NAMES},
            "candidate_extrinsics": {name: value.tolist() for name, value in candidate_extrinsics.items()},
        }
        (output / "relative_extrinsics_auto.yaml").write_text(
            yaml.safe_dump(candidate_doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        camera_doc = {
            "schema_version": 1,
            "state": "AUTO_CALIBRATION_VALID",
            "gauge": d0.GAUGE,
            "transforms": {name: value.tolist() for name, value in camera_transforms.items()},
            "closure": closure,
        }
        (output / "camera_extrinsics_auto.yaml").write_text(
            yaml.safe_dump(camera_doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        lut, lut_metrics = build_masked_lut(camera_transforms, geometry_path, mask_path)
        np.savez_compressed(output / "masked_bev_lut_auto.npz", **lut)
        report["camera_chain"] = closure
        report["lut"] = lut_metrics
    _write_trace(output / "iteration_trace.csv", trace)
    (output / "auto_calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if quality_pass else 2)


if __name__ == "__main__":
    main()
