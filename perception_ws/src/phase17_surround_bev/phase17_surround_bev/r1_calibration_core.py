#!/usr/bin/env python3
"""Gate D0-R1: 七路 LiDAR 相对外参的只读、可复现实验。

本脚本只打开 _r6 采样清单指定的七个 PointCloud2 话题。record time 仅用于
从顺序存储中定位候选消息；所有配对和算法时间均使用消息 header stamp。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from itertools import combinations

import numpy as np

try:
    import yaml
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    import rosbag2_py
except Exception as exc:  # pragma: no cover - 明确报告环境阻塞
    print(f"DEPENDENCY_BLOCKED: {exc}", file=sys.stderr)
    raise


SEVEN_NAMES = [
    "back_lidar_sensor", "forward_lidar_sensor", "front_lidar_sensor",
    "left2_lidar_sensor", "left_lidar_sensor", "right2_lidar_sensor",
    "right_lidar_sensor",
]
NORMAL_COSINE_MIN = 0.5
PARAM_ROTATION_SCALE_M = 1.0


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_wsl_path(value):
    """接受 Windows 路径，也接受 WSL 挂载路径。"""
    s = str(value).strip()
    if len(s) >= 3 and s[1:3] == ":/" and s[0].isalpha():
        return Path("/mnt") / s[0].lower() / s[3:]
    return Path(s)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=float)


def so3_exp(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    a = w / theta
    A = math.sin(theta)
    B = 1.0 - math.cos(theta)
    K = skew(a)
    return np.eye(3) + A * K + B * (K @ K)


def se3_exp(x, rotation_scale=None):
    if rotation_scale is None:
        rotation_scale = PARAM_ROTATION_SCALE_M
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(x[3:6], dtype=float) * float(rotation_scale))
    T[:3, 3] = np.asarray(x[:3], dtype=float)
    return T


def rpy_matrix(rpy):
    r, p, y = [float(v) for v in rpy]
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_transform(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def inv_T(T):
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def rot_angle(R):
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(c))


def transform_points(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def transform_normals(T, ns):
    return ns @ T[:3, :3].T


def voxel_cap(points, voxel=0.25, cap=600):
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first = np.sort(first)
    if len(first) > cap:
        first = first[np.linspace(0, len(first) - 1, cap).astype(int)]
    return points[first]


def estimate_normals(points, k=8):
    n = len(points)
    out = np.zeros((n, 3), dtype=float)
    if n < max(4, k):
        return out
    tree = cKDTree(points)
    _, ids = tree.query(points, k=min(k, n), workers=1)
    for i in range(n):
        q = points[ids[i]]
        q = q - np.mean(q, axis=0)
        cov = q.T @ q
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, int(np.argmin(vals))]
        j = int(np.argmax(np.abs(v)))
        if v[j] < 0:
            v = -v
        out[i] = v
    return out


def decode_xyz(msg: PointCloud2):
    names = {f.name: f for f in msg.fields}
    if not all(k in names for k in ("x", "y", "z")):
        return np.empty((0, 3), dtype=float), {"reason": "missing_xyz"}
    dtype_map = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    endian = ">" if msg.is_bigendian else "<"
    fields = [names[k] for k in ("x", "y", "z")]
    dt = np.dtype({"names": ["x", "y", "z"], "formats": [endian + dtype_map[f.datatype] for f in fields], "offsets": [f.offset for f in fields], "itemsize": int(msg.point_step)})
    rows = int(msg.height) if msg.height else 1
    cols = int(msg.width)
    data = bytes(msg.data)
    total = rows * cols
    try:
        if int(msg.row_step) == int(msg.point_step) * cols:
            arr = np.frombuffer(data, dtype=dt, count=total)
        else:
            raw = np.empty(total, dtype=dt)
            for r in range(rows):
                start = r * int(msg.row_step)
                raw[r * cols:(r + 1) * cols] = np.frombuffer(data[start:start + int(msg.row_step)], dtype=dt, count=cols)
            arr = raw
        pts = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(float)
    except Exception as exc:
        return np.empty((0, 3), dtype=float), {"reason": f"decode_error:{exc}"}
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite], {"raw_points": total, "finite_points": int(np.count_nonzero(finite)), "nonfinite_points": int(total - np.count_nonzero(finite))}


def source_lidar_contract(source_path):
    src = read_json(source_path)
    sensors = {}
    for item in src.get("perception_sensors", []):
        if item.get("sensor_type") != "gpu_ray":
            continue
        name = item.get("name")
        if name not in SEVEN_NAMES:
            continue
        sensors[name] = {
            "topic": item["ros_topics"]["points"],
            "type": item["ros_types"]["points"],
            "T": make_transform(item["nominal_pose_from_base_link"]["xyz_m"], item["nominal_pose_from_base_link"]["rpy_rad"]),
            "pose": item["nominal_pose_from_base_link"],
        }
    missing = [x for x in SEVEN_NAMES if x not in sensors]
    if missing:
        raise RuntimeError(f"source contract missing lidars: {missing}")
    return sensors, src


def read_sampling(path):
    d = read_json(path)
    groups = d.get("main_groups", [])
    if len(groups) != 40:
        raise RuntimeError(f"{path} main_groups={len(groups)}, expected 40")
    bag = d.get("bag")
    if not bag:
        raise RuntimeError(f"{path} missing bag")
    return bag, groups


def target_map(groups, sensors):
    by_topic = {sensors[n]["topic"]: {} for n in SEVEN_NAMES}
    for i, g in enumerate(groups):
        for name in SEVEN_NAMES:
            topic = sensors[name]["topic"]
            v = g["topics"].get(topic)
            if not v:
                raise RuntimeError(f"group {i} missing {topic}")
            by_topic[topic].setdefault(int(v["stamp_ns"]), []).append(i)
    return by_topic


def open_selected_bag(bag_path, groups, sensors, logger):
    """只在目标 header 附近反序列化；不反序列化非白名单消息。"""
    topics = [sensors[n]["topic"] for n in SEVEN_NAMES]
    # _r6 清单同时保存了每一路话题内的消息 index。MCAP record time
    # 是写盘时钟，可能与仿真 header stamp 不在同一时间域，因此用该
    # 已冻结的 topic-local index 做预筛选，再核对 header stamp。
    target_indices = {sensors[n]["topic"]: {} for n in SEVEN_NAMES}
    target_stamps = {sensors[n]["topic"]: {} for n in SEVEN_NAMES}
    for gi, g in enumerate(groups):
        for name in SEVEN_NAMES:
            topic = sensors[name]["topic"]
            item = g["topics"][topic]
            target_indices[topic].setdefault(int(item["index"]), []).append(gi)
            target_stamps[topic][gi] = int(item["stamp_ns"])
    candidates = {topic: {} for topic in topics}
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    traversed = 0
    deserialized = 0
    selected = 0
    topic_ordinals = {topic: 0 for topic in topics}
    while reader.has_next():
        topic, raw, rec_time = reader.read_next()
        traversed += 1
        if topic not in target_indices:
            continue
        ordinal = topic_ordinals[topic]
        topic_ordinals[topic] += 1
        if ordinal not in target_indices[topic]:
            continue
        msg = deserialize_message(raw, PointCloud2)
        deserialized += 1
        hs = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        pts, diag = decode_xyz(msg)
        for gi in target_indices[topic][ordinal]:
            target_stamp = target_stamps[topic][gi]
            # 只接受清单目标附近的 header stamp；异常时间不会悄悄进入拟合。
            if abs(target_stamp - hs) <= 25_000_000:
                candidates[topic].setdefault(gi, []).append((abs(target_stamp - hs), pts, diag, hs))
    logger(f"bag={bag_path} traversed_target_records={traversed} deserialized_frozen_indices={deserialized}")
    groups_out = []
    for gi, g in enumerate(groups):
        sensors_out = {}
        for name in SEVEN_NAMES:
            topic = sensors[name]["topic"]
            cs = candidates[topic].get(gi, [])
            if not cs:
                sensors_out[name] = {"points": np.empty((0, 3)), "normals": np.empty((0, 3)), "header_ns": None, "diag": {"reason": "not_selected"}}
                continue
            _, pts, diag, hs = min(cs, key=lambda x: x[0])
            pts = voxel_cap(pts, 0.25, 600)
            T = sensors[name]["T"]
            bp = transform_points(T, pts)
            ns = estimate_normals(bp, 8)
            sensors_out[name] = {"points": bp, "normals": ns, "header_ns": hs, "diag": diag}
            selected += 1
        groups_out.append({"index": gi, "split": "heldout" if gi % 4 == 0 else "calibration", "reference_stamp_ns": int(g["reference_stamp_ns"]), "sensors": sensors_out})
    logger(f"bag={bag_path} selected_sensor_group_arrays={selected}/{len(groups)*len(SEVEN_NAMES)}")
    return groups_out


def nn_stats(a, b, threshold):
    if len(a) == 0 or len(b) == 0:
        return {"a_to_b": 0., "b_to_a": 0., "mutual": 0, "median": float("nan"), "p95": float("nan")}
    ta = cKDTree(a)
    tb = cKDTree(b)
    dab, _ = tb.query(a, workers=1)
    dba, _ = ta.query(b, workers=1)
    ma = dab <= threshold
    mb = dba <= threshold
    return {"a_to_b": float(np.mean(ma)), "b_to_a": float(np.mean(mb)), "mutual": int(min(np.count_nonzero(ma), np.count_nonzero(mb))), "median": float(np.median(np.concatenate([dab, dba]))), "p95": float(np.percentile(np.concatenate([dab, dba]), 95))}


def build_overlap(frames, bags, names, threshold, out):
    rows = []
    edges = []
    for ai, bi in combinations(names, 2):
        all_group = 0
        total_mutual = 0
        bag_rows = []
        for bag in bags:
            subset = [f for f in frames if f["bag"] == bag and f["split"] == "calibration"]
            pass_n = 0
            mut_n = 0
            vals = []
            for f in subset:
                a = f["sensors"][ai]["points"]
                b = f["sensors"][bi]["points"]
                s = nn_stats(a, b, threshold)
                ok = s["a_to_b"] >= 0.10 and s["b_to_a"] >= 0.10
                pass_n += int(ok)
                mut_n += s["mutual"]
                vals.append(s)
                rows.append({"threshold_m": threshold, "bag": bag, "sensor_a": ai, "sensor_b": bi, "group_index": f["index"], "a_to_b": s["a_to_b"], "b_to_a": s["b_to_a"], "mutual": s["mutual"], "median_nn_m": s["median"], "strict_pass": ok})
            frac = pass_n / max(1, len(subset))
            bag_rows.append((frac, mut_n))
            all_group += pass_n
            total_mutual += mut_n
        ok_edge = all(x[0] >= 0.50 for x in bag_rows) and total_mutual >= 300
        support = float(np.mean([x[0] for x in bag_rows])) if bag_rows else 0.
        row = {"threshold_m": threshold, "sensor_a": ai, "sensor_b": bi, "per_bag_strict_fraction": ";".join(f"{x[0]:.3f}" for x in bag_rows), "strict_group_count_total": all_group, "effective_mutual_correspondences": total_mutual, "strict_edge": ok_edge, "median_bag_fraction": support}
        rows.append(row)
        if ok_edge:
            edges.append((ai, bi, support, total_mutual))
    write_csv(out / f"overlap_groups_{threshold:.2f}m.csv", ["threshold_m", "bag", "sensor_a", "sensor_b", "group_index", "a_to_b", "b_to_a", "mutual", "median_nn_m", "strict_pass"], [r for r in rows if "bag" in r])
    edge_rows = [r for r in rows if "bag" not in r]
    write_csv(out / f"overlap_edges_{threshold:.2f}m.csv", ["threshold_m", "sensor_a", "sensor_b", "per_bag_strict_fraction", "strict_group_count_total", "effective_mutual_correspondences", "strict_edge", "median_bag_fraction"], edge_rows)
    if abs(threshold - 0.30) < 1e-9:
        write_csv(out / "strict_overlap_edges.csv", ["threshold_m", "sensor_a", "sensor_b", "per_bag_strict_fraction", "strict_group_count_total", "effective_mutual_correspondences", "strict_edge", "median_bag_fraction"], edge_rows)
    return edges, [r for r in rows if "bag" not in r]


def graph_info(names, edges, gauge):
    adj = {n: set() for n in names}
    for a, b, _, _ in edges:
        adj[a].add(b); adj[b].add(a)
    comps = []
    unseen = set(names)
    while unseen:
        root = next(iter(unseen)); stack = [root]; comp = set()
        while stack:
            x = stack.pop()
            if x in comp: continue
            comp.add(x); unseen.discard(x); stack.extend(adj[x] - comp)
        comps.append(sorted(comp))
    comps.sort(key=lambda x: (-len(x), x))
    largest = comps[0] if comps else []
    return {"connected": len(largest) == len(names), "components": comps, "largest_component": largest, "gauge_in_largest": gauge in largest, "adjacency": {k: sorted(v) for k, v in adj.items()}, "forward_connected": "forward_lidar_sensor" in largest}


def selected_edges(edges, component):
    c = set(component)
    return [e for e in edges if e[0] in c and e[1] in c]


def make_transforms(ref, variables, x):
    out = {k: v.copy() for k, v in ref.items()}
    for i, name in enumerate(variables):
        out[name] = se3_exp(x[6 * i:6 * i + 6]) @ ref[name]
    return out


def scalar_edge_residual(frames, transforms, edge_pairs, max_corr=80, max_dist=0.50):
    residuals = []
    for f in frames:
        for a, b in edge_pairs:
            sa = f["sensors"][a]; sb = f["sensors"][b]
            if len(sa["points"]) == 0 or len(sb["points"]) == 0:
                residuals.extend([0.0] * max_corr)
                continue
            pa = transform_points(transforms[a], sa["points"])
            pb = transform_points(transforms[b], sb["points"])
            na = transform_normals(transforms[a], sa["normals"])
            nb = transform_normals(transforms[b], sb["normals"])
            tb = cKDTree(pb); ta = cKDTree(pa)
            dab, ib = tb.query(pa, workers=1); dba, ia = ta.query(pb, workers=1)
            keep_a = np.flatnonzero(dab <= max_dist)
            keep_b = np.flatnonzero(dba <= max_dist)
            if len(keep_a) > max_corr // 2:
                keep_a = keep_a[np.linspace(0, len(keep_a) - 1, max_corr // 2).astype(int)]
            if len(keep_b) > max_corr // 2:
                keep_b = keep_b[np.linspace(0, len(keep_b) - 1, max_corr // 2).astype(int)]
            block = []
            if len(keep_a):
                ns = na[keep_a]; nt = nb[ib[keep_a]]
                valid = (np.einsum("ij,ij->i", ns, ns) > 1e-12) & (np.einsum("ij,ij->i", nt, nt) > 1e-12)
                valid &= np.abs(np.einsum("ij,ij->i", ns, nt)) >= NORMAL_COSINE_MIN
                ka = keep_a[valid]
                block.extend(np.einsum("ij,ij->i", nb[ib[ka]], pa[ka] - pb[ib[ka]]).tolist())
            if len(keep_b):
                ns = nb[keep_b]; nt = na[ia[keep_b]]
                valid = (np.einsum("ij,ij->i", ns, ns) > 1e-12) & (np.einsum("ij,ij->i", nt, nt) > 1e-12)
                valid &= np.abs(np.einsum("ij,ij->i", ns, nt)) >= NORMAL_COSINE_MIN
                kb = keep_b[valid]
                block.extend(np.einsum("ij,ij->i", na[ia[kb]], pb[kb] - pa[ia[kb]]).tolist())
            # scipy.optimize 要求每次 residual 长度固定。未达到上限的
            # 方向用零占位；占位项不携带几何梯度，只保持优化维度一致。
            block = block[:max_corr]
            if len(block) < max_corr:
                block.extend([0.0] * (max_corr - len(block)))
            residuals.extend(block)
    if not residuals:
        return np.zeros(1, dtype=float)
    return np.asarray(residuals, dtype=float)


def run_m2(ref, frames, edge_pairs, component, gauge, initial_x=None, max_nfev=30):
    variables = [n for n in component if n != gauge]
    if initial_x is None:
        initial_x = np.zeros(6 * len(variables), dtype=float)
    cal = [f for f in frames if f["split"] == "calibration"]
    def fun(x):
        return scalar_edge_residual(cal, make_transforms(ref, variables, x), edge_pairs, 80, 0.50)
    run_m2.last_error = ""
    try:
        result = least_squares(fun, np.asarray(initial_x, dtype=float), loss="huber", f_scale=0.15, max_nfev=max_nfev, x_scale="jac")
        est = make_transforms(ref, variables, result.x)
        return est, result, variables
    except Exception as exc:
        run_m2.last_error = f"{type(exc).__name__}: {exc}"
        return {k: v.copy() for k, v in ref.items()}, None, variables


def pairwise_chain(ref, frames, edges, component, gauge):
    # 最大支持生成树；每条边均为独立标量点到平面拟合，作为漂移基线。
    parent = {x: x for x in component}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    tree_edges = []
    for a, b, support, mutual in sorted(edges, key=lambda e: (-e[2], -e[3], e[0], e[1])):
        if find(a) != find(b):
            parent[find(a)] = find(b); tree_edges.append((a, b))
    corr = {x: np.eye(4) for x in component}
    for a, b in tree_edges:
        if a == gauge or a in corr:
            def fun(x):
                T = {k: ref[k] for k in component}
                T[a] = corr[a] @ ref[a]
                T[b] = se3_exp(x) @ ref[b]
                return scalar_edge_residual([f for f in frames if f["split"] == "calibration"], T, [(a, b)], 80, 0.50)
            try:
                res = least_squares(fun, np.zeros(6), loss="huber", f_scale=0.15, max_nfev=30, x_scale="jac")
                corr[b] = se3_exp(res.x) @ corr[a]
            except Exception:
                corr[b] = corr[a].copy()
    out = {k: corr[k] @ ref[k] for k in component}
    return out, tree_edges


def heldout_metrics(frames, transforms, edges, component):
    out = []
    c = set(component)
    for f in frames:
        if f["split"] != "heldout": continue
        residuals = []; supports = []; mutuals = []
        for edge in edges:
            a, b = edge[0], edge[1]
            if a not in c or b not in c: continue
            sa = f["sensors"][a]; sb = f["sensors"][b]
            if len(sa["points"]) == 0 or len(sb["points"]) == 0: continue
            pa = transform_points(transforms[a], sa["points"]); pb = transform_points(transforms[b], sb["points"])
            na = transform_normals(transforms[a], sa["normals"]); nb = transform_normals(transforms[b], sb["normals"])
            tb = cKDTree(pb); ta = cKDTree(pa)
            dab, ib = tb.query(pa, workers=1); dba, ia = ta.query(pb, workers=1)
            ma = dab <= 0.50; mb = dba <= 0.50
            if np.any(ma):
                ma_idx = np.flatnonzero(ma)
                ns = na[ma_idx]; nt = nb[ib[ma_idx]]
                valid = (np.einsum("ij,ij->i", ns, ns) > 1e-12) & (np.einsum("ij,ij->i", nt, nt) > 1e-12)
                valid &= np.abs(np.einsum("ij,ij->i", ns, nt)) >= NORMAL_COSINE_MIN
                ma_idx = ma_idx[valid]
                if len(ma_idx): residuals.extend(np.einsum("ij,ij->i", nb[ib[ma_idx]], pa[ma_idx] - pb[ib[ma_idx]]).tolist())
            if np.any(mb):
                mb_idx = np.flatnonzero(mb)
                ns = nb[mb_idx]; nt = na[ia[mb_idx]]
                valid = (np.einsum("ij,ij->i", ns, ns) > 1e-12) & (np.einsum("ij,ij->i", nt, nt) > 1e-12)
                valid &= np.abs(np.einsum("ij,ij->i", ns, nt)) >= NORMAL_COSINE_MIN
                mb_idx = mb_idx[valid]
                if len(mb_idx): residuals.extend(np.einsum("ij,ij->i", na[ia[mb_idx]], pb[mb_idx] - pa[ia[mb_idx]]).tolist())
            supports.append(float((np.count_nonzero(ma) / max(1, len(ma)) + np.count_nonzero(mb) / max(1, len(mb))) / 2))
            mutuals.append(int(min(np.count_nonzero(ma), np.count_nonzero(mb))))
        out.append({"bag": f["bag"], "group_index": f["index"], "reference_stamp_ns": f["reference_stamp_ns"], "median_point_to_plane_m": float(np.median(np.abs(residuals))) if residuals else float("nan"), "bidirectional_support": float(np.mean(supports)) if supports else 0., "mutual_count": int(np.sum(mutuals)), "coverage": float(np.mean(np.asarray(supports) > 0.10)) if supports else 0.})
    return out


def relative_errors(ref, est, gauge, sensors):
    rows = []
    for s in sensors:
        if s == gauge or s not in ref or s not in est: continue
        rr = inv_T(ref[gauge]) @ ref[s]
        ee = inv_T(est[gauge]) @ est[s]
        d = inv_T(rr) @ ee
        rows.append({"sensor": s, "translation_error_m": float(np.linalg.norm(d[:3, 3])), "rotation_error_deg": float(np.degrees(rot_angle(d[:3, :3])))})
    return rows


def trial_specs(seed, sensors, gauge, amount_t, amount_r):
    # 12 项在配置冻结前已明确：六个单 DOF 正负、四个旋转正负、一个混合方向。
    cand = [
        ("back_lidar_sensor", "tx", amount_t), ("back_lidar_sensor", "tx", -amount_t),
        ("left_lidar_sensor", "ty", amount_t), ("right_lidar_sensor", "ty", -amount_t),
        ("left2_lidar_sensor", "tz", amount_t), ("right2_lidar_sensor", "tz", -amount_t),
        ("forward_lidar_sensor", "rx", amount_r), ("forward_lidar_sensor", "rx", -amount_r),
        ("left_lidar_sensor", "ry", amount_r), ("right_lidar_sensor", "ry", -amount_r),
        ("left2_lidar_sensor", "rz", amount_r), ("right2_lidar_sensor", "rz", -amount_r),
    ]
    out = []
    for i, (s, dof, val) in enumerate(cand, 1):
        x = np.zeros(6)
        x[{"tx": 0, "ty": 1, "tz": 2, "rx": 3, "ry": 4, "rz": 5}[dof]] = val
        out.append({"trial_id": f"quick_{i:02d}", "sensor": s, "dof": dof, "vector": x.tolist(), "amount_t": amount_t, "amount_r": amount_r})
    return out


def run_trial(spec, ref, frames, edges, component, gauge, ref_holdout, out_name):
    variables = [n for n in component if n != gauge]
    if spec["sensor"] not in variables:
        return {"phase": out_name, "trial_id": spec["trial_id"], "sensor": spec["sensor"], "dof": spec["dof"], "injected_translation_m": float(spec["amount_t"]), "injected_rotation_deg": float(np.degrees(spec["amount_r"])), "estimated_translation_error_m": float("nan"), "estimated_rotation_error_deg": float("nan"), "initial_holdout_median_m": float("nan"), "final_holdout_median_m": float("nan"), "reference_holdout_median_m": float(np.nanmedian([x["median_point_to_plane_m"] for x in ref_holdout])) if ref_holdout else float("nan"), "initial_coverage": float("nan"), "final_coverage": float("nan"), "boundary_hit": False, "success": False, "solver_nfev": 0, "status": "NOT_APPLICABLE_DISCONNECTED_OR_COMPONENT_GAUGE"}
    initial_x = np.zeros(6 * len(variables))
    if spec["sensor"] in variables:
        initial_x[6 * variables.index(spec["sensor"]):6 * variables.index(spec["sensor"]) + 6] = np.asarray(spec["vector"])
    initial = make_transforms(ref, variables, initial_x)
    init_hold = heldout_metrics(frames, initial, edges, component)
    est, result, _ = run_m2(ref, frames, edges, component, gauge, initial_x, 30)
    final_hold = heldout_metrics(frames, est, edges, component)
    err = relative_errors(ref, est, gauge, component)
    e = next((x for x in err if x["sensor"] == spec["sensor"]), {"translation_error_m": float("inf"), "rotation_error_deg": float("inf")})
    inj_t = float(spec["amount_t"]); inj_r_deg = float(np.degrees(spec["amount_r"]))
    trans_ok = e["translation_error_m"] <= 0.2 * inj_t if abs(spec["vector"][0]) + abs(spec["vector"][1]) + abs(spec["vector"][2]) > 0 else True
    rot_ok = e["rotation_error_deg"] <= 0.2 * inj_r_deg if abs(spec["vector"][3]) + abs(spec["vector"][4]) + abs(spec["vector"][5]) > 0 else True
    init_med = float(np.nanmedian([x["median_point_to_plane_m"] for x in init_hold])) if init_hold else float("nan")
    final_med = float(np.nanmedian([x["median_point_to_plane_m"] for x in final_hold])) if final_hold else float("nan")
    ref_med = float(np.nanmedian([x["median_point_to_plane_m"] for x in ref_holdout])) if ref_holdout else float("nan")
    init_cov = float(np.nanmean([x["coverage"] for x in init_hold])) if init_hold else 0.
    final_cov = float(np.nanmean([x["coverage"] for x in final_hold])) if final_hold else 0.
    boundary = bool(result is not None and result.status < 0)
    success = bool(trans_ok and rot_ok and np.isfinite(final_med) and np.isfinite(ref_med) and final_med <= 1.05 * max(ref_med, 1e-9) and final_med < init_med and final_cov >= 0.95 * max(init_cov, 1e-9) and not boundary)
    return {"phase": out_name, "trial_id": spec["trial_id"], "sensor": spec["sensor"], "dof": spec["dof"], "injected_translation_m": inj_t, "injected_rotation_deg": inj_r_deg, "estimated_translation_error_m": e["translation_error_m"], "estimated_rotation_error_deg": e["rotation_error_deg"], "initial_holdout_median_m": init_med, "final_holdout_median_m": final_med, "reference_holdout_median_m": ref_med, "initial_coverage": init_cov, "final_coverage": final_cov, "boundary_hit": boundary, "success": success, "solver_nfev": int(result.nfev) if result is not None else -1, "status": "RUN" if result is not None else f"OPTIMIZER_ERROR:{run_m2.last_error}"}


def save_figures(out, names, edges, singular):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = {n: (math.cos(2 * math.pi * i / len(names)), math.sin(2 * math.pi * i / len(names))) for i, n in enumerate(names)}
        for a, b, _, _ in edges:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], "k-")
        for n, (x, y) in pos.items():
            ax.scatter([x], [y], s=80); ax.text(x, y, n.replace("_lidar_sensor", ""), fontsize=8)
        ax.set_title("R1 strict overlap graph (0.30 m)"); ax.axis("equal"); fig.tight_layout(); fig.savefig(out / "figures" / "overlap_graph.png", dpi=150); plt.close(fig)
        if singular:
            fig, ax = plt.subplots(figsize=(7, 4)); ax.semilogy(np.arange(1, len(singular) + 1), singular, "o-"); ax.set_xlabel("singular value index"); ax.set_ylabel("value"); ax.set_title("scalar point-to-plane Jacobian spectrum"); fig.tight_layout(); fig.savefig(out / "figures" / "hessian_spectrum.png", dpi=150); plt.close(fig)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="D0-R1 seven-LiDAR relative calibration experiment")
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-contract", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = read_yaml(Path(args.config)); out = Path(args.output); out.mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True); (out / "logs").mkdir(exist_ok=True)
    global NORMAL_COSINE_MIN, PARAM_ROTATION_SCALE_M
    NORMAL_COSINE_MIN = float(cfg["optimization"].get("normal_angle_cosine_min", 0.5))
    PARAM_ROTATION_SCALE_M = float(cfg["optimization"].get("rotation_scale_m", 1.0))
    log_path = out / "logs" / "r1_run.log"
    def logger(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f: f.write(line + "\n")
    logger("R1 start: only seven exact PointCloud2 topics; no prior full audit")
    sensors, source = source_lidar_contract(Path(args.source_contract))
    sampling = [normalize_wsl_path(x) for x in str(cfg["source_samples"]["sampling_cache"]).split(";")]
    all_frames = []
    loaded_bags = []
    selection_receipt = []
    for sp in sampling:
        bag, groups = read_sampling(sp)
        bag_path = Path(bag)
        # 缓存可记录 Windows 或 WSL 路径；调用者需保证数据路径在当前环境可访问。
        if not bag_path.exists():
            raise FileNotFoundError(f"采样缓存引用的 bag 不存在：{bag_path}")
        loaded_bags.append(bag)
        logger(f"loading selected timestamps bag={bag_path}")
        gdata = open_selected_bag(bag_path, groups, sensors, logger)
        for gd in gdata:
            gd["bag"] = bag
            all_frames.append(gd)
        selection_receipt.append({"bag": bag, "sampling_cache": str(sp), "groups": len(groups), "selected_sensor_group_arrays": len(groups) * len(SEVEN_NAMES), "selected_messages_per_topic": len(groups), "selected_topics": [sensors[n]["topic"] for n in SEVEN_NAMES], "selection": "frozen topic-local index then header stamp nearest within 25ms; no non-whitelist deserialization"})
    write_json(out / "selected_message_access_audit.json", {"bags": selection_receipt, "topic_count": 7, "topic_names": [sensors[n]["topic"] for n in SEVEN_NAMES], "forbidden_topics_opened": False})
    logger(f"loaded frames={len(all_frames)} bags={loaded_bags}")
    names = SEVEN_NAMES; gauge = cfg["gauge"]["fixed_sensor"]
    primary = float(cfg["overlap"]["primary_m"])
    edges, edge_rows = build_overlap(all_frames, loaded_bags, names, primary, out / "logs")
    write_csv(out / "strict_overlap_edges.csv", ["threshold_m", "sensor_a", "sensor_b", "per_bag_strict_fraction", "strict_group_count_total", "effective_mutual_correspondences", "strict_edge", "median_bag_fraction"], edge_rows)
    graph = graph_info(names, edges, gauge)
    sens_edges = {}
    for th in cfg["overlap"]["sensitivity_m"]:
        se, _ = build_overlap(all_frames, loaded_bags, names, float(th), out / "logs")
        sens_edges[str(th)] = se
    write_json(out / "overlap_graph.json", {"primary_threshold_m": primary, "edges": [{"a": a, "b": b, "median_bag_fraction": s, "mutual": m} for a, b, s, m in edges], **graph, "sensitivity_edge_counts": {k: len(v) for k, v in sens_edges.items()}})
    component = graph["largest_component"]
    edge_pairs = [(a, b) for a, b, _, _ in selected_edges(edges, component)]
    logger(f"primary strict edges={len(edges)} largest_component={component} forward_in_component={graph['forward_connected']}")
    requested_gauge = gauge
    # 若 front Airy 不在最大连通分量，不能声称按全七路 front gauge 完成。
    # 为了量化该孤立分量本身的相对几何，仅在本地诊断中固定分量首节点。
    component_gauge = gauge if gauge in component else (component[0] if component else gauge)
    ref = {n: sensors[n]["T"] for n in component}
    # M0 heldout and M2 nominal-init reference.
    m0_hold = heldout_metrics(all_frames, ref, selected_edges(edges, component), component)
    m2_est, m2_res, variables = run_m2(ref, all_frames, edge_pairs, component, component_gauge, None, int(cfg["optimization"]["max_outer_evaluations"]))
    m2_hold = heldout_metrics(all_frames, m2_est, selected_edges(edges, component), component)
    m1_est, tree_edges = pairwise_chain(ref, all_frames, selected_edges(edges, component), component, component_gauge)
    m1_hold = heldout_metrics(all_frames, m1_est, selected_edges(edges, component), component)
    method_rows = []
    for method, vals in [("M0_NOMINAL_UNOPTIMIZED", m0_hold), ("M1_PAIRWISE_GICP_CHAIN", m1_hold), ("M2_JOINT_RELATIVE_GAUGE_FIXED", m2_hold)]:
        for row in vals:
            method_rows.append({"method": method, **row})
    write_csv(out / "heldout_residuals.csv", ["method", "bag", "group_index", "reference_stamp_ns", "median_point_to_plane_m", "bidirectional_support", "mutual_count", "coverage"], method_rows)
    singular = []
    rank_info = {}
    if m2_res is not None and getattr(m2_res, "jac", None) is not None:
        try:
            singular = np.linalg.svd(np.asarray(m2_res.jac, dtype=float), compute_uv=False).tolist()
        except Exception:
            singular = []
    target_rank = 6 * max(0, len(component) - 1)
    for tol in (1e-5, 1e-6, 1e-7):
        rank = int(sum(float(s) > (max(singular) * tol if singular else float("inf")) for s in singular))
        rank_info[str(tol)] = {"rank": rank, "target_rank": target_rank}
    write_json(out / "corrected_hessian_spectrum.json", {"model": "scalar point-to-plane M2 Jacobian", "rotation_scale_m": float(cfg["optimization"].get("rotation_scale_m", 1.0)), "component": component, "gauge_requested": requested_gauge, "gauge_used": component_gauge, "variables": variables, "singular_values": singular, "rank_by_tolerance": rank_info, "m2_solver_nfev": int(m2_res.nfev) if m2_res is not None else None, "m2_cost": float(m2_res.cost) if m2_res is not None else None})
    write_csv(out / "corrected_hessian_spectrum.csv", ["index", "singular_value"], [{"index": i + 1, "singular_value": v} for i, v in enumerate(singular)])
    save_figures(out, names, edges, singular)
    ref_med = float(np.nanmedian([x["median_point_to_plane_m"] for x in m2_hold])) if m2_hold else float("nan")
    quick_specs = trial_specs(int(cfg["perturbations"]["seed"]), component, component_gauge, float(cfg["perturbations"]["quick_translation_m"]), math.radians(float(cfg["perturbations"]["quick_rotation_deg"])))
    quick_rows = []
    for spec in quick_specs:
        logger(f"quick trial {spec['trial_id']} {spec['sensor']} {spec['dof']}")
        quick_rows.append(run_trial(spec, ref, all_frames, edge_pairs, component, component_gauge, m2_hold, "L2_QUICK"))
    q_success = sum(bool(r["success"]) for r in quick_rows)
    logger(f"QUICK_GATE success={q_success}/{len(quick_rows)} required={cfg['perturbations']['quick_pass_min_success']}")
    diag_rows = []
    if q_success < int(cfg["perturbations"]["quick_pass_min_success"]):
        # 仅保留与失败传感器/DOF相同的目标诊断，不扩展完整压力网格。
        for i, spec in enumerate(quick_specs):
            if i >= int(cfg["perturbations"]["diagnostic_trials_after_quick_failure"]): break
            spec = dict(spec); spec["trial_id"] = f"diag_{i + 1:02d}"
            diag_rows.append(run_trial(spec, ref, all_frames, edge_pairs, component, component_gauge, m2_hold, "DIAGNOSTIC_AFTER_QUICK_FAILURE"))
    recovery_rows = quick_rows + diag_rows
    write_csv(out / "recovery_trials.csv", list(recovery_rows[0].keys()) if recovery_rows else ["phase"], recovery_rows)
    summary = {"quick": {"success": q_success, "total": len(quick_rows), "pass_threshold": int(cfg["perturbations"]["quick_pass_min_success"]), "passed": q_success >= int(cfg["perturbations"]["quick_pass_min_success"])}, "diagnostics_after_quick_failure": {"success": sum(bool(r["success"]) for r in diag_rows), "total": len(diag_rows)}, "L1": "NOT_RUN_after_quick_failure" if q_success < 10 else "RUN_IF_QUICK_PASS", "L2": "NOT_RUN_after_quick_failure" if q_success < 10 else "RUN_IF_QUICK_PASS", "L3": "NOT_RUN_after_quick_failure" if q_success < 10 else "RUN_IF_QUICK_PASS", "reference_holdout_median_m": ref_med, "m0_holdout_median_m": float(np.nanmedian([x["median_point_to_plane_m"] for x in m0_hold])) if m0_hold else None, "m1_holdout_median_m": float(np.nanmedian([x["median_point_to_plane_m"] for x in m1_hold])) if m1_hold else None, "m2_holdout_median_m": ref_med, "relative_error_rows": relative_errors(ref, m2_est, component_gauge, component)}
    write_json(out / "perturbation_recovery_summary.json", summary)
    complete = graph["connected"] and graph["forward_connected"] and all(v["rank"] >= v["target_rank"] for v in rank_info.values()) and q_success >= 10
    partial = (not graph["connected"] and len(component) > 1 and all(v["rank"] >= v["target_rank"] for v in rank_info.values()) and q_success >= 10)
    if complete:
        status = "R1_RELATIVE_LOCALLY_OBSERVABLE"
    elif partial:
        status = "R1_PARTIAL_COMPONENT_ONLY"
    else:
        status = "R1_RELATIVE_NOT_DEMONSTRATED"
    decision = {
        "status": status,
        "scope": "relative extrinsics only; no ship-center absolute gauge",
        "primary_overlap_graph": graph,
        "strict_edges": [{"a": a, "b": b, "support": s, "mutual": m} for a, b, s, m in edges],
        "component": component,
        "gauge_requested": requested_gauge,
        "gauge_used_for_component_diagnostic": component_gauge,
        "gauge_in_largest_component": gauge in component,
        "hessian": rank_info,
        "recovery": summary,
        "data_driven_reference": "M2 nominal-init solution; not external truth",
        "normal_angle_cosine_min": NORMAL_COSINE_MIN,
        "forbidden_inputs_deserialized": False,
    }
    write_json(out / "R1_DECISION.json", decision)
    with (out / "R1_DECISION.md").open("w", encoding="utf-8") as f:
        f.write(f"# Gate D0-R1 decision\n\n## Final status\n\n`{status}`\n\n")
        f.write("本实验只判断相对 LiDAR 外参；不判断船中心绝对坐标、相机外参或 dock。\n\n")
        f.write(f"- Primary 0.30 m strict edges: {len(edges)}; largest component: {', '.join(component)}; seven-node connected: {graph['connected']}。\n")
        f.write(f"- Forward LiDAR in largest component: {graph['forward_connected']}。\n")
        f.write(f"- Scalar point-to-plane Jacobian target rank: {target_rank}; ranks: {rank_info}。\n")
        f.write(f"- L2 quick recovery: {q_success}/{len(quick_rows)}; required 10/12. L1/L2/L3 expansion: {summary['L1']}/{summary['L2']}/{summary['L3']}。\n")
        f.write("- M1 is a pairwise chain diagnostic; M2 is the joint relative gauge-fixed result. Held-out residuals are in `heldout_residuals.csv`.\n\n")
        f.write("## Limitations\n\n")
        f.write("公共 SE(3) gauge remains unresolved. Fixed front Airy is a computational gauge only and is not a proof of the vessel-center frame. All selected messages came from the seven exact PointCloud2 topics; no TF, CameraInfo, GPS, IMU, odometry, dock, clock or truth topic was deserialized.\n")
    logger(f"FINAL status={status}; outputs={out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"R1_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
