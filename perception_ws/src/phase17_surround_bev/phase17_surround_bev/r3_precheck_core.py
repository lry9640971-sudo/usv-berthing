#!/usr/bin/env python3
"""读取一份 R3 algorithm_input MCAP，生成冻结 0.30 m 预检指标。"""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


NAMES = ["back_lidar_sensor", "forward_lidar_sensor", "front_lidar_sensor", "left2_lidar_sensor", "left_lidar_sensor", "right2_lidar_sensor", "right_lidar_sensor"]
BRIDGES = [("forward_lidar_sensor", "front_lidar_sensor"), ("front_lidar_sensor", "right_lidar_sensor"), ("left2_lidar_sensor", "back_lidar_sensor")]


def load_r1(path):
    spec = importlib.util.spec_from_file_location("r1_reference", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def load_sensors(path, r1):
    sensors, _ = r1.source_lidar_contract(Path(path))
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in raw["perception_sensors"]:
        if item.get("name") in sensors and item.get("sensor_type") == "gpu_ray":
            sensors[item["name"]]["lidar"] = item["lidar"]
    return sensors


def transform_points(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def voxel(points, size=0.20):
    if len(points) == 0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def native_filter(points, sensor):
    if len(points) == 0:
        return points
    lidar = sensor["lidar"]; distance = np.linalg.norm(points, axis=1); horizontal = np.linalg.norm(points[:, :2], axis=1)
    azimuth = np.arctan2(points[:, 1], points[:, 0]); elevation = np.arctan2(points[:, 2], np.maximum(horizontal, 1e-9))
    amin = float(lidar["horizontal_min_angle_rad"]); amax = float(lidar["horizontal_max_angle_rad"])
    horizontal_ok = (azimuth >= amin) & (azimuth <= amax) if amin <= amax else ((azimuth >= amin) | (azimuth <= amax))
    return points[(distance >= float(lidar["min_range_m"])) & (distance <= float(lidar["max_range_m"])) & horizontal_ok & (elevation >= float(lidar["vertical_min_angle_rad"])) & (elevation <= float(lidar["vertical_max_angle_rad"]))]


def metric(a, b, radius=0.30):
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0, 0, float("nan")
    db, _ = cKDTree(b).query(a, workers=1); da, _ = cKDTree(a).query(b, workers=1)
    ma = db <= radius; mb = da <= radius
    mutual = min(int(ma.sum()), int(mb.sum()))
    return float(ma.mean()), float(mb.mean()), mutual, float(np.median(np.r_[db[ma], da[mb]])) if mutual else float("nan")


def normal_groups(points_a, points_b, r1, separation_deg=20.0):
    """以局部 PCA 法向的无向夹角统计确定性方向组。"""
    vectors = []
    for points in (points_a, points_b):
        sample = voxel(points, 0.50)
        if len(sample) > 400:
            sample = sample[np.linspace(0, len(sample) - 1, 400).astype(int)]
        if len(sample) >= 8:
            normals = r1.estimate_normals(sample, k=8)
            vectors.extend(normals[np.linalg.norm(normals, axis=1) > 0.5])
    groups = []
    cos_limit = math.cos(math.radians(separation_deg))
    for vector in vectors:
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if all(abs(float(np.dot(vector, item))) < cos_limit for item in groups):
            groups.append(vector)
    return len(groups)


def read_bag(bag, sensors, r1):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    topics = [sensors[n]["topic"] for n in NAMES]
    rows = {t: [] for t in topics}
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        msg = deserialize_message(raw, PointCloud2)
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        points, _diag = r1.decode_xyz(msg)
        rows[topic].append((stamp, points))
    if min(len(v) for v in rows.values()) < 10:
        raise RuntimeError("R3_PRECHECK_BLOCKED: one or more exact lidar topics have fewer than 10 messages")
    return rows


def main():
    parser = argparse.ArgumentParser(description="R3 single-pose LiDAR calibration arena precheck")
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--r1-script", required=True)
    parser.add_argument("--input-bag", required=True, help="algorithm_input MCAP only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=40)
    args = parser.parse_args()
    bag = Path(args.input_bag); out = Path(args.output)
    if not bag.exists():
        raise SystemExit(f"R3_PRECHECK_BLOCKED: input MCAP not found: {bag}")
    out.mkdir(parents=True, exist_ok=True)
    r1 = load_r1(args.r1_script); sensors = load_sensors(Path(args.source_contract), r1); records = read_bag(bag, sensors, r1)
    ref_topic = sensors["forward_lidar_sensor"]["topic"]
    refs = records[ref_topic]
    idx = np.linspace(0, len(refs) - 1, min(args.groups, len(refs))).astype(int)
    fields = ["group", "reference_stamp_ns", "edge", "a_to_b", "b_to_a", "mutual_voxel", "median_distance_m", "n_a_voxel", "n_b_voxel", "raw_mutual_a", "raw_mutual_b", "raw_mutual_pair", "normal_group_count", "formal_pass"]
    rows_out = []
    for group, ri in enumerate(idx):
        stamp = refs[int(ri)][0]; selected = {}
        for name in NAMES:
            topic = sensors[name]["topic"]; nearest = min(records[topic], key=lambda x: abs(x[0] - stamp))
            if abs(nearest[0] - stamp) > 25_000_000:
                raise RuntimeError(f"R3_PRECHECK_BLOCKED: sync gap >25ms at group {group}, {name}")
            local = native_filter(nearest[1], sensors[name]); raw = transform_points(sensors[name]["T"], local); selected[name] = (raw, voxel(raw))
        for a, b in BRIDGES:
            raw_a, va = selected[a]; raw_b, vb = selected[b]
            sa, sb, mutual, residual = metric(va, vb)
            rsa, rsb, raw_mutual, _rr = metric(raw_a, raw_b)
            raw_a_count = int(round(rsa * len(raw_a)))
            raw_b_count = int(round(rsb * len(raw_b)))
            normal_count = normal_groups(raw_a, raw_b, r1)
            rows_out.append({"group": group, "reference_stamp_ns": stamp, "edge": f"{a}__{b}", "a_to_b": sa, "b_to_a": sb, "mutual_voxel": mutual, "median_distance_m": residual, "n_a_voxel": len(va), "n_b_voxel": len(vb), "raw_mutual_a": raw_a_count, "raw_mutual_b": raw_b_count, "raw_mutual_pair": raw_mutual, "normal_group_count": normal_count, "formal_pass": bool(sa >= .10 and sb >= .10 and mutual >= 100 and raw_mutual >= 300 and normal_count >= 3)})
    with (out / "precheck_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows_out)
    print(f"R3 precheck wrote {len(rows_out)} bridge rows to {out / 'precheck_metrics.csv'}")


if __name__ == "__main__":
    main()
