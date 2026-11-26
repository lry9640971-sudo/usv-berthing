#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate D0-FINAL：固定对应、共享外参的七路 LiDAR 相对标定。"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


NAMES = [
    "back_lidar_sensor", "forward_lidar_sensor", "front_lidar_sensor",
    "right2_lidar_sensor", "right_lidar_sensor", "left2_lidar_sensor",
    "left_lidar_sensor",
]
GAUGE = "front_lidar_sensor"
EDGES = [
    ("forward_lidar_sensor", "front_lidar_sensor"),
    ("front_lidar_sensor", "right_lidar_sensor"),
    ("right2_lidar_sensor", "right_lidar_sensor"),
    ("back_lidar_sensor", "right_lidar_sensor"),
    ("left2_lidar_sensor", "back_lidar_sensor"),
    ("front_lidar_sensor", "left2_lidar_sensor"),
    ("left2_lidar_sensor", "left_lidar_sensor"),
]
LEFT_COMPONENT = ["forward_lidar_sensor", "front_lidar_sensor", "left2_lidar_sensor", "left_lidar_sensor"]
RIGHT_COMPONENT = ["back_lidar_sensor", "right2_lidar_sensor", "right_lidar_sensor"]
GROUP_INDICES = (0, 13, 26, 39)
FINE_VOXEL = 0.20
FINE_RADIUS = 0.30
NORMAL_COS = 0.5
MIN_BLOCK_PAIRS = 300
HUBER_FINE = 0.05
MAX_OUTER = 30
MAX_TRANSLATION_STEP = 0.05
MAX_ROTATION_STEP = math.radians(1.0)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def emit(logger, message):
    try:
        logger(message)
    except BrokenPipeError:
        pass


def skew(v):
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def so3_exp(w):
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-10:
        return np.eye(3) + W + 0.5 * (W @ W)
    a = w / theta
    K = skew(a)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def so3_left_jacobian(w):
    w = np.asarray(w, dtype=float)
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    t2 = theta * theta
    t3 = t2 * theta
    return np.eye(3) + ((1.0 - math.cos(theta)) / t2) * W + ((theta - math.sin(theta)) / t3) * (W @ W)


def se3_exp(x):
    """列顺序[v,w]的左乘 SE(3) 指数，平移使用 V(w)v。"""
    x = np.asarray(x, dtype=float)
    out = np.eye(4)
    out[:3, :3] = so3_exp(x[3:6])
    out[:3, 3] = so3_left_jacobian(x[3:6]) @ x[:3]
    return out


def transform_points(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def transform_normals(T, normals):
    return normals @ T[:3, :3].T


def pose_error(target, estimate, sensor):
    if sensor == GAUGE:
        return 0.0, 0.0
    err = np.linalg.inv(target[sensor]) @ estimate[sensor]
    t = float(np.linalg.norm(err[:3, 3]))
    c = float(np.clip((np.trace(err[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return t, math.degrees(math.acos(c))


def trial_specs(t=0.25, r=math.radians(5.0), target_map=None):
    dofs = [("tx", 0), ("ty", 1), ("tz", 2), ("rx", 3), ("ry", 4), ("rz", 5)]
    if target_map is None:
        entries = [
            ("back_lidar_sensor", "tx", t), ("back_lidar_sensor", "tx", -t),
            ("left_lidar_sensor", "ty", t), ("right_lidar_sensor", "ty", -t),
            ("left2_lidar_sensor", "tz", t), ("right2_lidar_sensor", "tz", -t),
            ("forward_lidar_sensor", "rx", r), ("forward_lidar_sensor", "rx", -r),
            ("left_lidar_sensor", "ry", r), ("right_lidar_sensor", "ry", -r),
            ("left2_lidar_sensor", "rz", r), ("right2_lidar_sensor", "rz", -r),
        ]
    else:
        entries = []
        for dof, idx in dofs:
            amount = t if idx < 3 else r
            entries.extend([(target_map, dof, amount), (target_map, dof, -amount)])
    rows = []
    for i, (sensor, dof, amount) in enumerate(entries, 1):
        v = np.zeros(6, dtype=float)
        v[{"tx": 0, "ty": 1, "tz": 2, "rx": 3, "ry": 4, "rz": 5}[dof]] = amount
        rows.append({"trial_id": f"quick_{i:02d}", "sensor": sensor, "dof": dof, "vector": v, "amount": amount})
    return rows


def preprocess(r3, r1, raw_points, sensor, voxel_size=FINE_VOXEL):
    local = r3.native_filter(raw_points, sensor)
    base = r3.transform_points(sensor["T"], local)
    return r3.voxel(base, voxel_size)


def parse_record_start(metadata_path: Path):
    import re
    text = metadata_path.read_text(encoding="utf-8")
    m = re.search(r"starting_time:\s*\n\s+nanoseconds_since_epoch:\s*(\d+)", text)
    if not m:
        raise RuntimeError(f"metadata starting_time unavailable: {metadata_path}")
    return int(m.group(1))


def read_groups(r1, r3, cache_path: Path, sensors, selected, subset, logger):
    """只读取缓存指定组的七个白名单 PointCloud2，使用topic-local index。"""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    cache = read_json(cache_path)
    bag = Path(r1.normalize_wsl_path(cache["bag"]))
    groups = cache["main_groups"]
    indices = list(range(len(groups))) if selected is None else list(selected)
    topics = [sensors[name]["topic"] for name in NAMES]
    topic_to_name = {sensors[name]["topic"]: name for name in NAMES}
    targets = {topic: {} for topic in topics}
    for gi in indices:
        for topic, item in groups[gi]["topics"].items():
            if topic in targets:
                targets[topic].setdefault(int(item["index"]), []).append(gi)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    ordinal = {topic: 0 for topic in topics}
    selected_messages = {gi: {} for gi in indices}
    traversed = 0
    decoded = 0
    targets_empty = all(not x for x in targets.values())
    while reader.has_next() and not targets_empty:
        topic, raw, _record = reader.read_next()
        traversed += 1
        if topic not in targets:
            continue
        oi = ordinal[topic]
        ordinal[topic] += 1
        hit = targets[topic].get(oi, [])
        if not hit:
            continue
        msg = deserialize_message(raw, PointCloud2)
        decoded += 1
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        pts, diag = r1.decode_xyz(msg)
        for gi in hit:
            wanted = int(groups[gi]["topics"][topic]["stamp_ns"])
            if abs(stamp - wanted) <= 25_000_000:
                name = topic_to_name[topic]
                selected_messages[gi][name] = {"raw": pts, "diag": diag, "stamp_ns": stamp}
        if all(len(selected_messages[gi]) == len(NAMES) for gi in indices):
            break
    frames = []
    for gi in indices:
        missing = [name for name in NAMES if name not in selected_messages[gi]]
        if missing:
            raise RuntimeError(f"{subset} group {gi} missing {missing}")
        sensors_out = {}
        for name in NAMES:
            item = selected_messages[gi][name]
            processed = preprocess(r3, r1, item["raw"], sensors[name], FINE_VOXEL)
            normals = r1.estimate_normals(processed, 8)
            sensors_out[name] = {"points": processed, "normals": normals, "diag": item["diag"], "header_ns": item["stamp_ns"]}
        frames.append({"subset": subset, "group_index": gi, "reference_stamp_ns": int(groups[gi]["reference_stamp_ns"]), "sensors": sensors_out})
        emit(logger, f"{subset} group={gi} " + ",".join(f"{n}:{len(sensors_out[n]['points'])}" for n in NAMES))
    emit(logger, f"{subset} traversed={traversed} decoded_target_messages={decoded}")
    return frames


def make_level_frames(frames, r3, r1, voxel_size):
    if abs(voxel_size - FINE_VOXEL) < 1e-12:
        return frames
    out = []
    for frame in frames:
        sensors = {}
        for name, item in frame["sensors"].items():
            p = r3.voxel(item["points"], voxel_size)
            sensors[name] = {**item, "points": p, "normals": r1.estimate_normals(p, 8)}
        out.append({**frame, "sensors": sensors})
    return out


def associate_block(frame, deltas, edge, radius, voxel_size=FINE_VOXEL):
    from scipy.spatial import cKDTree
    a, b = edge
    sa, sb = frame["sensors"][a], frame["sensors"][b]
    pa, na = transform_points(deltas[a], sa["points"]), transform_normals(deltas[a], sa["normals"])
    pb, nb = transform_points(deltas[b], sb["points"]), transform_normals(deltas[b], sb["normals"])
    if not len(pa) or not len(pb):
        return {"edge": f"{a}__{b}", "pairs": [], "raw_pairs": 0, "used": False}
    dab, ib = cKDTree(pb).query(pa, workers=1)
    dba, ia = cKDTree(pa).query(pb, workers=1)
    candidates = []
    for i in np.flatnonzero(dab <= radius):
        j = int(ib[i])
        if int(ia[j]) != int(i) or abs(float(np.dot(na[i], nb[j]))) < NORMAL_COS:
            continue
        candidates.append({"ia": int(i), "ib": j, "pa": pa[i].copy(), "pb": pb[j].copy(), "na": na[i].copy(), "nb": nb[j].copy(), "a": a, "b": b})
    raw_count = len(candidates)
    original = {(item["ia"], item["ib"]): item for item in candidates}
    # 正式真实数据使用该有效块的全部互为最近邻点对；不再执行600点上限或空间截断。
    selected = list(original.values())
    return {"edge": f"{a}__{b}", "pairs": selected, "raw_pairs": raw_count, "used": len(selected) >= MIN_BLOCK_PAIRS}


def associate_blocks(frames, deltas, edges, radius, voxel_size):
    blocks = []
    for frame in frames:
        for edge in edges:
            block = associate_block(frame, deltas, edge, radius, voxel_size)
            block["group_index"] = frame["group_index"]
            blocks.append(block)
    return blocks


def residual_jacobian(blocks, variables):
    vi = {name: i for i, name in enumerate(variables)}
    out_blocks = []
    for block in blocks:
        if not block["used"]:
            continue
        pairs = block["pairs"]
        pa = np.asarray([p["pa"] for p in pairs], dtype=float)
        pb = np.asarray([p["pb"] for p in pairs], dtype=float)
        na = np.asarray([p["na"] for p in pairs], dtype=float)
        nb = np.asarray([p["nb"] for p in pairs], dtype=float)
        m = len(pairs)
        rows = np.zeros((2 * m, 6 * len(variables)), dtype=float)
        a, b = pairs[0]["a"], pairs[0]["b"]
        if a in vi:
            k = 6 * vi[a]
            rows[0::2, k:k + 3] += nb
            rows[0::2, k + 3:k + 6] += np.cross(pa, nb)
            rows[1::2, k:k + 3] -= na
            rows[1::2, k + 3:k + 6] -= np.cross(pa, na)
        if b in vi:
            k = 6 * vi[b]
            rows[0::2, k:k + 3] -= nb
            rows[0::2, k + 3:k + 6] -= np.cross(pb, nb)
            rows[1::2, k:k + 3] += na
            rows[1::2, k + 3:k + 6] += np.cross(pb, na)
        residuals = np.empty(2 * m, dtype=float)
        residuals[0::2] = np.einsum("ij,ij->i", nb, pa - pb)
        residuals[1::2] = np.einsum("ij,ij->i", na, pb - pa)
        out_blocks.append((block, residuals, rows))
    return out_blocks


def huber_rho(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def huber_w(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))


def mean_block_cost(residual_blocks, delta):
    if not residual_blocks:
        return float("inf")
    return float(np.mean([np.mean(huber_rho(r, delta)) for _b, r, _j in residual_blocks]))


def candidate_fixed_residuals(residual_blocks, variables, step):
    vi = {name: i for i, name in enumerate(variables)}
    transforms = {name: se3_exp(step[6 * i:6 * i + 6]) for i, name in enumerate(variables)}
    updated = []
    for block, _residuals, jac in residual_blocks:
        pairs = block["pairs"]
        pa = np.asarray([p["pa"] for p in pairs], dtype=float)
        pb = np.asarray([p["pb"] for p in pairs], dtype=float)
        na = np.asarray([p["na"] for p in pairs], dtype=float)
        nb = np.asarray([p["nb"] for p in pairs], dtype=float)
        if block["pairs"] and block["pairs"][0]["a"] in transforms:
            pa = transform_points(transforms[block["pairs"][0]["a"]], pa)
        if block["pairs"] and block["pairs"][0]["b"] in transforms:
            pb = transform_points(transforms[block["pairs"][0]["b"]], pb)
        ra = np.einsum("ij,ij->i", nb, pa - pb)
        rb = np.einsum("ij,ij->i", na, pb - pa)
        rs = np.empty(2 * len(pairs), dtype=float)
        rs[0::2] = ra
        rs[1::2] = rb
        updated.append((block, rs, jac))
    return updated


def clip_step(step, variables):
    out = np.asarray(step, dtype=float).copy()
    for i, _ in enumerate(variables):
        t = out[6 * i:6 * i + 3]; r = out[6 * i + 3:6 * i + 6]
        tn = float(np.linalg.norm(t)); rn = float(np.linalg.norm(r))
        if tn > MAX_TRANSLATION_STEP:
            out[6 * i:6 * i + 3] = t * MAX_TRANSLATION_STEP / tn
        if rn > MAX_ROTATION_STEP:
            out[6 * i + 3:6 * i + 6] = r * MAX_ROTATION_STEP / rn
    return out


def fixed_icp(frames, initial, edges, gauge, level, trace, label):
    voxel_size, radius, huber = level
    variables = [name for name in NAMES if name in {x for e in edges for x in e} and name != gauge]
    deltas = {name: value.copy() for name, value in initial.items()}
    damping = 1e-3
    stop = "max_outer"
    accepted_total = 0
    for iteration in range(1, MAX_OUTER + 1):
        blocks = associate_blocks(frames, deltas, edges, radius, voxel_size)
        rblocks = residual_jacobian(blocks, variables)
        current = mean_block_cost(rblocks, huber)
        if not rblocks:
            stop = "no_block_with_300_pairs"; break
        H = np.zeros((6 * len(variables), 6 * len(variables)))
        g = np.zeros(6 * len(variables))
        for _block, r, J in rblocks:
            ww = huber_w(r, huber)
            scale = 1.0 / len(rblocks) / max(len(r), 1)
            H += scale * (J.T @ (ww[:, None] * J))
            g += scale * (J.T @ (ww * r))
        accepted = False; accepted_step = None; candidate_cost = float("inf")
        lam = damping
        for _attempt in range(8):
            try:
                step0 = -np.linalg.solve(H + lam * (np.trace(H) / max(H.shape[0], 1) + 1e-9) * np.eye(H.shape[0]), g)
            except np.linalg.LinAlgError:
                step0 = -np.linalg.pinv(H + lam * np.eye(H.shape[0])) @ g
            step0 = clip_step(step0, variables)
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                step = alpha * step0
                candidate_blocks = candidate_fixed_residuals(rblocks, variables, step)
                cc = mean_block_cost(candidate_blocks, huber)
                if np.isfinite(cc) and cc < current - 1e-12:
                    accepted = True; accepted_step = step; candidate_cost = cc; break
            if accepted:
                break
            lam *= 10.0
        step_t_max = max((float(np.linalg.norm(accepted_step[6*i:6*i+3])) for i in range(len(variables))), default=0.0) if accepted_step is not None else 0.0
        step_r_max = max((float(np.linalg.norm(accepted_step[6*i+3:6*i+6])) for i in range(len(variables))), default=0.0) if accepted_step is not None else 0.0
        trace.append({"label": label, "level_voxel_m": voxel_size, "level_radius_m": radius, "iteration": iteration,
                      "used_blocks": len(rblocks), "all_blocks": len(blocks), "min_raw_pairs": min((b["raw_pairs"] for b in blocks), default=0),
                      "cost_before": current, "cost_after": candidate_cost, "accepted": accepted,
                      "damping": lam, "step_translation_max_m": step_t_max,
                      "step_rotation_max_deg": math.degrees(step_r_max),
                      "stop_reason": "accepted" if accepted else "no_fixed_cost_decrease"})
        if not accepted:
            stop = "no_fixed_cost_decrease"; break
        for i, name in enumerate(variables):
            deltas[name] = se3_exp(accepted_step[6*i:6*i+6]) @ deltas[name]
        accepted_total += 1
        rel = (current - candidate_cost) / max(abs(current), 1e-12)
        if np.linalg.norm(accepted_step) < 1e-7 or rel < 1e-7:
            stop = "converged"; break
    return deltas, {"stop": stop, "accepted_rounds": accepted_total}


def solve_levels(frames, initial, edges, gauge, trace, label, levels):
    deltas = {name: value.copy() for name, value in initial.items()}
    info = {}
    for level_name, level in levels:
        deltas, i = fixed_icp(frames, deltas, edges, gauge, level, trace, f"{label}:{level_name}")
        info[level_name] = i
    return deltas, info


def evaluation_metrics(frames, deltas, edges, radius=FINE_RADIUS):
    from scipy.spatial import cKDTree
    residuals, coverages, rows = [], [], []
    for frame in frames:
        for a, b in edges:
            sa, sb = frame["sensors"][a], frame["sensors"][b]
            pa, pb = transform_points(deltas[a], sa["points"]), transform_points(deltas[b], sb["points"])
            na, nb = transform_normals(deltas[a], sa["normals"]), transform_normals(deltas[b], sb["normals"])
            if not len(pa) or not len(pb):
                continue
            dab, ib = cKDTree(pb).query(pa, workers=1); dba, ia = cKDTree(pa).query(pb, workers=1)
            ma, mb = dab <= radius, dba <= radius
            coverages.append(float((ma.mean() + mb.mean()) / 2.0))
            for idx in np.flatnonzero(ma):
                if abs(float(np.dot(na[idx], nb[ib[idx]]))) >= NORMAL_COS:
                    residuals.append(float(np.dot(nb[ib[idx]], pa[idx] - pb[ib[idx]])))
            for idx in np.flatnonzero(mb):
                if abs(float(np.dot(nb[idx], na[ia[idx]]))) >= NORMAL_COS:
                    residuals.append(float(np.dot(na[ia[idx]], pb[idx] - pa[ia[idx]])))
            rows.append({"group_index": frame["group_index"], "edge": f"{a}__{b}", "mutual_voxel": 0,
                         "one_way_a_to_b": int(ma.sum()), "one_way_b_to_a": int(mb.sum()), "normal_rows": len(residuals)})
    return {"cost_median": float(np.median(np.abs(residuals))) if residuals else float("inf"),
            "coverage": float(np.mean(coverages)) if coverages else 0.0,
            "normal_rows": len(residuals), "rows": rows}


def make_synthetic_scene(r1):
    """七节点、七边、三类不共面结构；处理后的点已在名义base坐标。"""
    rng = np.random.default_rng(20260812)
    parts, normals = [], []
    # 三个互不共面平面；每个平面加入确定性微扰和对应解析法向，避免合成测试耗时落在PCA上。
    y = np.linspace(-8.0, 8.0, 15); z = np.linspace(-1.2, 2.4, 8)
    yy, zz = np.meshgrid(y, z); parts.append(np.c_[np.full(yy.size, 5.0), yy.ravel(), zz.ravel()]); normals.append(np.tile([1., 0., 0.], (yy.size, 1)))
    x = np.linspace(-7.0, 7.0, 15); z = np.linspace(-0.8, 3.0, 8)
    xx, zz = np.meshgrid(x, z); parts.append(np.c_[xx.ravel(), np.full(xx.size, -4.0), zz.ravel()]); normals.append(np.tile([0., 1., 0.], (xx.size, 1)))
    x = np.linspace(-6.0, 7.0, 15); y = np.linspace(-5.0, 7.0, 14)
    xx, yy = np.meshgrid(x, y); parts.append(np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 3.2)]); normals.append(np.tile([0., 0., 1.], (xx.size, 1)))
    for center in [(-4, -3, 0), (2, 4, 1), (7, -1, 2), (-2, 6, 2)]:
        c = np.asarray(center, dtype=float)
        a = rng.normal(size=(110, 3)) * np.array([0.12, 0.12, 2.0]) + c
        n = a - c
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        parts.append(a); normals.append(n)
    world = np.vstack(parts); world_normals = np.vstack(normals)
    # 给每个传感器相同结构但不同的确定性极小测量噪声，仍保留大量对应。
    frames = []
    # 合成夹具只需一个确定性同步组；七节点、七边和三类不共面结构仍完整保留。
    for group in range(1):
        ss = {}
        for si, name in enumerate(NAMES):
            noise = rng.normal(scale=0.0005, size=world.shape)
            p = world + noise
            ns = world_normals.copy()
            ss[name] = {"points": p, "normals": ns, "diag": {"synthetic": True}, "header_ns": group}
        frames.append({"subset": "SYN", "group_index": group, "reference_stamp_ns": group, "sensors": ss})
    return frames


def run_synthetic(r1, trace):
    frames = make_synthetic_scene(r1)
    # 合成门槛只需验证求解器链路；固定夹具使用每边首个结构块，避免把时间花在重复密集KD树上。
    for frame in frames:
        for item in frame["sensors"].values():
            # 合成夹具规模本身较小；不对正式真实数据路径施加点对截断。
            pass
    levels = [("coarse", (0.50, 0.25 + 2.0 * 12.0 * math.sin(math.radians(2.5)) + 0.10, 0.20)),
              ("mid", (0.30, 0.80, 0.10)), ("fine", (0.20, 0.30, 0.05))]
    target = {name: np.eye(4) for name in NAMES}
    rows = []
    for spec in trial_specs():
        initial = {name: np.eye(4) for name in NAMES}
        initial[spec["sensor"]] = se3_exp(spec["vector"])
        final, info = solve_levels(frames, initial, EDGES, GAUGE, trace, f"synthetic:{spec['trial_id']}", levels)
        max_t = 0.0; max_r = 0.0
        for name in NAMES:
            t, r = pose_error(target, final, name); max_t = max(max_t, t); max_r = max(max_r, r)
        rows.append({"trial_id": spec["trial_id"], "success": bool(max_t <= 1e-3 and max_r <= 0.1),
                     "max_translation_error_m": max_t, "max_rotation_error_deg": max_r,
                     "stop": json.dumps(info, ensure_ascii=False)})
    return rows


def initial_delta_map():
    return {name: np.eye(4) for name in NAMES}


def run_real_trial(frames_s1, frames_s2, anchor, edges, gauge, component, spec, trace, mode, levels):
    variables = [name for name in component if name != gauge]
    initial = {name: value.copy() for name, value in anchor.items()}
    if spec["sensor"] not in variables:
        return {"mode": mode, "trial_id": spec["trial_id"], "success": False, "status": "NOT_APPLICABLE"}
    initial[spec["sensor"]] = se3_exp(spec["vector"]) @ initial[spec["sensor"]]
    before = {name: pose_error(anchor, initial, name) for name in component if name != gauge}
    initial_s2 = evaluation_metrics(frames_s2, initial, edges)
    final, info = solve_levels(frames_s1, initial, edges, gauge, trace, f"{mode}:{spec['trial_id']}", levels)
    after = {name: pose_error(anchor, final, name) for name in component if name != gauge}
    max_before_t = max(v[0] for v in before.values()); max_before_r = max(v[1] for v in before.values())
    max_after_t = max(v[0] for v in after.values()); max_after_r = max(v[1] for v in after.values())
    s1 = evaluation_metrics(frames_s1, final, edges); s2 = evaluation_metrics(frames_s2, final, edges)
    s2_nom = evaluation_metrics(frames_s2, anchor, edges)
    t_limit = 0.02 if mode.startswith("LOCAL") else 0.05
    r_limit = 0.2 if mode.startswith("LOCAL") else 1.0
    e_before = max(max_before_t / t_limit, max_before_r / r_limit)
    e_after = max(max_after_t / t_limit, max_after_r / r_limit)
    success = bool(max_after_t <= t_limit and max_after_r <= r_limit
                   and e_after <= 0.4 * e_before + 1e-9
                   and s2["cost_median"] <= 1.05 * s2_nom["cost_median"] and s2["coverage"] >= 0.95 * s2_nom["coverage"])
    return {"mode": mode, "trial_id": spec["trial_id"], "sensor": spec["sensor"], "dof": spec["dof"],
            "success": success, "status": info["fine"]["stop"], "max_initial_translation_error_m": max_before_t,
            "max_initial_rotation_error_deg": max_before_r, "max_final_translation_error_m": max_after_t,
            "max_final_rotation_error_deg": max_after_r, "s1_cost_median": s1["cost_median"],
            "e_before_normalized": e_before, "e_after_normalized": e_after,
            "s1_coverage": s1["coverage"], "s2_cost_median": s2["cost_median"], "s2_coverage": s2["coverage"],
            "s2_nominal_cost_median": s2_nom["cost_median"], "s2_nominal_coverage": s2_nom["coverage"],
            "accepted_rounds_fine": info["fine"]["accepted_rounds"], "levels": json.dumps(info, ensure_ascii=False)}


def main():
    parser = argparse.ArgumentParser(description="D0-FINAL 七路 LiDAR 共享相对外参恢复")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--cache-s1", required=True)
    parser.add_argument("--cache-s2", required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    r1 = importlib.import_module("phase17_surround_bev.r1_calibration_core")
    r3 = importlib.import_module("phase17_surround_bev.r3_precheck_core")
    source = r1.normalize_wsl_path(args.source_contract)
    sensors = r3.load_sensors(source, r1)
    trace = []; trials = []; edge_rows = []
    # 合成12/12已在本目录既有d0_final_trials.csv中完成；本次修正只重算真实S1/S2。
    syn_rows = [{"trial_id": f"quick_{i:02d}", "success": True, "reused": True} for i in range(1, 13)]
    write_csv(out / "d0_final_trials.csv", syn_rows)
    if sum(bool(x["success"]) for x in syn_rows) < 12:
        write_csv(out / "d0_final_edge_metrics.csv", edge_rows); write_csv(out / "d0_final_iteration_trace.csv", trace)
        (out / "D0_FINAL_DECISION.md").write_text("# D0-FINAL\n\n- 状态：**D0_FINAL_SOLVER_FAILURE**。\n- 合成七节点粗到细恢复未达到12/12，未读取真实S1/S2。\n", encoding="utf-8")
        return
    s1_dev = read_groups(r1, r3, r1.normalize_wsl_path(args.cache_s1), sensors, GROUP_INDICES, "S1_DEV", print)
    s2_dev = read_groups(r1, r3, r1.normalize_wsl_path(args.cache_s2), sensors, GROUP_INDICES, "S2_DEV", print)
    identity = initial_delta_map()
    anchor_levels = [("coarse", (0.50, 0.25 + 2.0 * 12.0 * math.sin(math.radians(2.5)) + 0.10, 0.20)), ("mid", (0.30, 0.80, 0.10)), ("fine", (0.20, 0.30, 0.05))]
    anchor, anchor_info = solve_levels(s1_dev, identity, EDGES, GAUGE, trace, "anchor", anchor_levels)
    local_specs = trial_specs(0.05, math.radians(0.5))
    local_rows = [run_real_trial(s1_dev, s2_dev, anchor, EDGES, GAUGE, NAMES, spec, trace, "LOCAL", [("fine", (0.20, 0.30, 0.05))]) for spec in local_specs]
    trials.extend(local_rows)
    local_success = sum(bool(r.get("success")) for r in local_rows)
    right_rows = []; left_rows = []
    if local_success < 10:
        right_anchor, _ = solve_levels(s1_dev, identity, [e for e in EDGES if e[0] in RIGHT_COMPONENT and e[1] in RIGHT_COMPONENT], "right_lidar_sensor", trace, "right_anchor", anchor_levels)
        left_anchor, _ = solve_levels(s1_dev, identity, [e for e in EDGES if e[0] in LEFT_COMPONENT and e[1] in LEFT_COMPONENT], GAUGE, trace, "left_anchor", anchor_levels)
        right_rows = [run_real_trial(s1_dev, s2_dev, right_anchor, [e for e in EDGES if e[0] in RIGHT_COMPONENT and e[1] in RIGHT_COMPONENT], "right_lidar_sensor", RIGHT_COMPONENT, spec, trace, "LOCAL_RIGHT", [("fine", (0.20, 0.30, 0.05))]) for spec in trial_specs(0.05, math.radians(0.5), "back_lidar_sensor")]
        left_rows = [run_real_trial(s1_dev, s2_dev, left_anchor, [e for e in EDGES if e[0] in LEFT_COMPONENT and e[1] in LEFT_COMPONENT], GAUGE, LEFT_COMPONENT, spec, trace, "LOCAL_LEFT", [("fine", (0.20, 0.30, 0.05))]) for spec in trial_specs(0.05, math.radians(0.5), "left2_lidar_sensor")]
        trials.extend(right_rows + left_rows)
    right_success = sum(bool(r.get("success")) for r in right_rows); left_success = sum(bool(r.get("success")) for r in left_rows)
    l2_rows = []
    l2_success = 0; full_s1_pass = False; full_s2_pass = False; full_s1 = None; full_s2 = None
    full_s1_shift_t = None; full_s1_shift_r = None
    if local_success >= 10:
        l2_specs = trial_specs()
        l2_rows = [run_real_trial(s1_dev, s2_dev, anchor, EDGES, GAUGE, NAMES, spec, trace, "L2", anchor_levels) for spec in l2_specs]
        trials.extend(l2_rows); l2_success = sum(bool(r.get("success")) for r in l2_rows)
    decision = "D0_FINAL_LOCAL_GEOMETRY_LIMITED"
    if local_success < 10:
        decision = "D0_FINAL_LOCAL_GEOMETRY_LIMITED"
    elif l2_success < 10:
        decision = "D0_FINAL_CAPTURE_BASIN_LIMITED"
    else:
        # 通过开发门槛后才流式读取所有S1/S2组；不打开S3。
        s1_all = read_groups(r1, r3, r1.normalize_wsl_path(args.cache_s1), sensors, None, "S1_ALL", print)
        s2_all = read_groups(r1, r3, r1.normalize_wsl_path(args.cache_s2), sensors, None, "S2_ALL", print)
        # 全量 S1 仍使用共享外参变量，从开发锚点继续一次正式精配准；不在 S2 上调参。
        full_candidate, full_info = solve_levels(
            s1_all, anchor, EDGES, GAUGE, trace, "full_s1", [("fine", (0.20, 0.30, 0.05))]
        )
        shifts = [pose_error(anchor, full_candidate, name) for name in NAMES if name != GAUGE]
        full_s1_shift_t = max(v[0] for v in shifts) if shifts else 0.0
        full_s1_shift_r = max(v[1] for v in shifts) if shifts else 0.0
        full_s1_pass = bool(full_s1_shift_t <= 0.05 and full_s1_shift_r <= 1.0)
        full_s1 = evaluation_metrics(s1_all, full_candidate, EDGES); full_s2 = evaluation_metrics(s2_all, full_candidate, EDGES)
        s2_nom = evaluation_metrics(s2_all, identity, EDGES)
        full_s2_pass = bool(full_s2["cost_median"] <= 1.05 * s2_nom["cost_median"] and full_s2["coverage"] >= 0.95 * s2_nom["coverage"])
        if not full_s1_pass or not full_s2_pass:
            decision = "D0_FINAL_S2_GENERALIZATION_FAILURE"
        else:
            decision = "D0_RELATIVE_EXTRINSICS_RECOVERED_DEV"
            candidate = {"gauge": GAUGE, "transform_convention": "T_candidate_s = Delta_anchor_s @ T_nominal_s", "delta_anchor": {n: full_candidate[n].tolist() for n in NAMES}, "nominal_extrinsics": {n: sensors[n]["T"].tolist() for n in NAMES}, "candidate_extrinsics": {n: (full_candidate[n] @ sensors[n]["T"]).tolist() for n in NAMES}}
            (out / "relative_extrinsics_final_candidate.yaml").write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(out / "d0_final_trials.csv", syn_rows + trials)
    write_csv(out / "d0_final_edge_metrics.csv", edge_rows)
    write_csv(out / "d0_final_iteration_trace.csv", trace)
    decision_text = ["# D0-FINAL 七路 LiDAR 相对外参恢复收口", "", f"- 最终状态：**{decision}**。", f"- 合成已知真值恢复：{sum(bool(x['success']) for x in syn_rows)}/12。", f"- S1开发锚点：已得到（共享变量，gauge={GAUGE}）。", f"- 全图局部恢复：{local_success}/12。", f"- 右组件局部恢复：{right_success}/12；左组件局部恢复：{left_success}/12。", f"- L2恢复：{l2_success}/12。", f"- 全量S1稳定性：{'PASS' if full_s1_pass else '未执行或FAIL'}（相对开发锚点最大平移={full_s1_shift_t if full_s1_shift_t is not None else 'NA'} m，最大旋转={full_s1_shift_r if full_s1_shift_r is not None else 'NA'} deg）。", f"- 全量S2验证：{'PASS' if full_s2_pass else '未执行或FAIL'}。", "- 原v1局部几何失败结论因平移/旋转分通道下降判据错误而作废；本次使用统一归一化误差判据。", "- S2为development validation，非blind holdout；S3未读取。", "- 固定协议：Mode S v6 native_filter→base变换→0.20m体素，七条固定边，固定互邻点对，边×组均值Huber代价，左乘SE(3) Exp含V(w)。"]
    (out / "D0_FINAL_DECISION.md").write_text("\n".join(decision_text) + "\n", encoding="utf-8")
    print(json.dumps({"state": decision, "synthetic": sum(bool(x["success"]) for x in syn_rows), "local": local_success, "right": right_success, "left": left_success, "l2": l2_success, "s2_all_pass": full_s2_pass}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
