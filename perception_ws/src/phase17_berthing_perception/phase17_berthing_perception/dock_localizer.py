#!/usr/bin/env python3
"""Phase16Z Dock truth-free 多假设、多分辨率局部配准器。"""

from __future__ import annotations

import csv
import math
import time
from typing import Iterable

import numpy as np

try:
    from scipy.signal import fftconvolve
    from scipy.spatial import cKDTree
except Exception:
    fftconvolve = None
    cKDTree = None


MODEL_BAYS = {1: (0.0, 3.0), 2: (0.0, 9.0), 3: (0.0, 15.0)}
X_MIN, X_MAX = 4.0, 28.0
Y_MIN, Y_MAX = -22.0, 10.0
COARSE_RES = 0.20
TRUNC = 1.50
SUPPORT = 0.45
VISIBLE_MARGIN = 1.00
BEAM_WIDTH = 6
REFINE_SCHEDULE = ((1.0, 4.0), (0.25, 1.0), (0.05, 0.25))
PRIOR_REFINE_SCHEDULE = ((2.0, 8.0), (0.5, 2.0), (0.1, 0.5), (0.025, 0.1))


def load_template(path):
    points = []
    groups = {"PORT_FINGER": [], "STARBOARD_FINGER": [], "BACK_WALL": []}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            point = [float(row["x_m"]), float(row["y_m"])]
            points.append(point)
            groups[row["structure_group"]].append(point)
    return np.asarray(points, dtype=np.float64), {
        name: np.asarray(value, dtype=np.float64) for name, value in groups.items()
    }


def rotate(points: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    matrix = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ matrix.T


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def robust_trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return TRUNC
    clipped = np.minimum(values, TRUNC)
    keep = max(1, int(math.ceil(len(clipped) * (1.0 - trim_fraction))))
    return float(np.mean(np.sort(clipped)[:keep]))


def coarse_domain_grid():
    xs = np.arange(X_MIN, X_MAX + COARSE_RES * 0.5, COARSE_RES, dtype=np.float64)
    ys = np.arange(Y_MIN, Y_MAX + COARSE_RES * 0.5, COARSE_RES, dtype=np.float64)
    yaws = np.arange(-30.0, 30.0 + 0.5, 1.0, dtype=np.float64)
    return xs, ys, np.deg2rad(yaws)


def occupancy_grid(points: np.ndarray) -> np.ndarray:
    ncols = int(round((X_MAX - X_MIN) / COARSE_RES)) + 1
    nrows = int(round((Y_MAX - Y_MIN) / COARSE_RES)) + 1
    grid = np.zeros((nrows, ncols), dtype=np.float64)
    if len(points):
        ix = np.rint((points[:, 0] - X_MIN) / COARSE_RES).astype(int)
        iy = np.rint((Y_MAX - points[:, 1]) / COARSE_RES).astype(int)
        valid = (ix >= 0) & (ix < ncols) & (iy >= 0) & (iy < nrows)
        grid[iy[valid], ix[valid]] = 1.0
    return grid


def full_fft_candidates(points: np.ndarray, template: np.ndarray) -> list[tuple[float, float, float, float]]:
    """用完整域相关产生候选；这里只做粗定位，不承担最终精确审计。"""

    observed = occupancy_grid(points)
    candidates = []
    for yaw_deg in np.arange(-30.0, 30.1, 1.0):
        yaw = math.radians(float(yaw_deg))
        transformed = rotate(template, yaw)
        ix = np.rint(transformed[:, 0] / COARSE_RES).astype(int)
        iy = np.rint(-transformed[:, 1] / COARSE_RES).astype(int)
        kernel = np.zeros(
            (max(1, int(np.ptp(iy)) + 1), max(1, int(np.ptp(ix)) + 1)),
            dtype=np.float64,
        )
        kernel[iy - iy.min(), ix - ix.min()] = 1.0
        padded = np.zeros(
            (observed.shape[0] + kernel.shape[0] - 1, observed.shape[1] + kernel.shape[1] - 1),
            dtype=np.float64,
        )
        padded[: observed.shape[0], : observed.shape[1]] = observed
        if fftconvolve is None:
            correlation = np.zeros_like(padded)
        else:
            correlation = fftconvolve(padded, kernel[::-1, ::-1], mode="same")
        keep = min(16, correlation.size)
        for flat_index in np.argpartition(correlation.ravel(), -keep)[-keep:]:
            row, column = np.unravel_index(flat_index, correlation.shape)
            tx = X_MIN + (column - kernel.shape[1] // 2) * COARSE_RES - transformed[:, 0].min()
            ty = Y_MAX - (row - kernel.shape[0] // 2) * COARSE_RES - transformed[:, 1].max()
            if X_MIN <= tx <= X_MAX and Y_MIN <= ty <= Y_MAX:
                candidates.append((float(correlation[row, column]), float(tx), float(ty), yaw))
    candidates.sort(reverse=True)
    return candidates[:256]


def nms(candidates: Iterable[tuple[float, float, float, float]], limit: int = 16):
    kept = []
    for candidate in sorted(candidates, reverse=True):
        if any(
            math.hypot(candidate[1] - other[1], candidate[2] - other[2]) < 0.8
            and abs(math.degrees(wrap_angle(candidate[3] - other[3]))) < 2.0
            for other in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


class FrameScorer:
    """在一帧上共享 KD-tree，避免多假设优化反复构造索引。"""

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        self.tree = cKDTree(self.points) if cKDTree is not None and len(self.points) else None
        self._model_id = None
        self._model_tree = None
        self._group_indices = None

    def score(self, template: np.ndarray, groups: dict, pose: tuple[float, float, float]) -> dict:
        tx, ty, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        transformed = rotate(template, yaw) + np.asarray([tx, ty], dtype=np.float64)
        if self._model_id != id(template) and cKDTree is not None and len(template):
            self._model_id = id(template)
            self._model_tree = cKDTree(template)
            self._group_indices = {name: self._model_tree.query(group)[1] for name, group in groups.items()}
        if self.tree is None or not len(transformed):
            return {
                "x": tx,
                "y": ty,
                "yaw": wrap_angle(yaw),
                "score": TRUNC,
                "t2o": TRUNC,
                "o2t": TRUNC,
                "active_groups": 0,
                "total_support": 0,
                "support_normalized": 0.0,
                "visibility_fraction": 0.0,
                "visible_group_count": 0,
                "residual": TRUNC,
                "support": {},
            }

        template_distances = np.asarray(self.tree.query(transformed, k=1)[0], dtype=np.float64)
        t2o = robust_trimmed_mean(template_distances, 0.35)
        bounds_min = transformed.min(axis=0) - 1.5
        bounds_max = transformed.max(axis=0) + 1.5
        in_roi = np.all((self.points >= bounds_min) & (self.points <= bounds_max), axis=1)
        local = self.points[in_roi]
        if len(local):
            # 刚体变换不改变距离；在固定模板树上查询，避免每个候选反复建树。
            canonical = rotate(local - (tx, ty), -yaw)
            observed_distances = np.asarray(self._model_tree.query(canonical, k=1)[0], dtype=np.float64)
            o2t = robust_trimmed_mean(observed_distances, 0.65)
        else:
            o2t = TRUNC

        support = {}
        visible_coverages = []
        active_groups = 0
        total_support = 0
        for name, group in groups.items():
            indices = self._group_indices[name]
            transformed_group = transformed[indices]
            distances = template_distances[indices]
            group_min = transformed_group.min(axis=0) - VISIBLE_MARGIN
            group_max = transformed_group.max(axis=0) + VISIBLE_MARGIN
            local_group = np.all((self.points >= group_min) & (self.points <= group_max), axis=1)
            expected_visible = int(np.count_nonzero(local_group)) >= max(4, int(0.015 * len(group)))
            coverage = float(np.count_nonzero(distances <= SUPPORT) / max(1, len(group)))
            if expected_visible:
                visible_coverages.append(coverage)
                if coverage >= 0.15:
                    active_groups += 1
                total_support += int(np.count_nonzero(distances <= SUPPORT))
            support[name] = {
                "visible": bool(expected_visible),
                "coverage": coverage,
                "count": int(np.count_nonzero(distances <= SUPPORT)),
                "expected_points": len(group),
            }

        if visible_coverages:
            support_normalized = float(np.mean(visible_coverages))
            visibility_fraction = float(len(visible_coverages) / max(1, len(groups)))
        else:
            support_normalized = 0.0
            visibility_fraction = 0.0
        support_penalty = 1.0 - support_normalized
        visibility_penalty = 1.0 - visibility_fraction
        score = (
            0.30 * min(t2o, TRUNC)
            + 0.25 * min(o2t, TRUNC)
            + 0.30 * support_penalty
            + 0.15 * visibility_penalty
        )
        return {
            "x": tx,
            "y": ty,
            "yaw": wrap_angle(yaw),
            "score": float(score),
            "t2o": float(t2o),
            "o2t": float(o2t),
            "active_groups": int(active_groups),
            "total_support": int(total_support),
            "support_normalized": support_normalized,
            "visibility_fraction": visibility_fraction,
            "visible_group_count": len(visible_coverages),
            "residual": float(min(TRUNC, 0.5 * (t2o + o2t))),
            "support": support,
        }


def score_pose(points, template, groups, pose):
    """兼容旧 evaluator 的单帧接口。"""

    return FrameScorer(points).score(template, groups, pose)


def hypothesis_status(rows: list[dict], gap: float = 0.02) -> str:
    if not rows:
        return "NO_ESTIMATE"
    ordered = sorted(rows, key=lambda item: float(item["score"]))
    if len(ordered) > 1 and float(ordered[1]["score"]) - float(ordered[0]["score"]) < gap:
        return "AMBIGUOUS"
    return "UNIQUE"


def transition_penalty(candidate: dict, reference: dict) -> float:
    """对静态 Dock 的帧间跳变做无真值惩罚。"""

    distance = math.hypot(
        float(candidate["x"]) - float(reference["x"]),
        float(candidate["y"]) - float(reference["y"]),
    )
    yaw_jump = abs(math.degrees(wrap_angle(float(candidate["yaw"]) - float(reference["yaw"]))))
    return float(min(1.0, distance / 2.0) + min(1.0, yaw_jump / 5.0))


def apply_temporal_prior(rows: list[dict], reference: dict | None, weight: float) -> list[dict]:
    if reference is None:
        return sorted(rows, key=lambda item: float(item["score"]))
    output = []
    for row in rows:
        item = dict(row)
        item["geometry_score"] = float(row["score"])
        item["transition_penalty"] = transition_penalty(row, reference)
        item["score"] = float(item["geometry_score"]) + float(weight) * float(item["transition_penalty"])
        output.append(item)
    return sorted(output, key=lambda item: float(item["score"]))


def apply_coarse_pose_prior(
    rows: list[dict],
    prior: dict,
    *,
    position_deadband_m: float = 1.5,
    yaw_deadband_deg: float = 5.0,
    position_gate_m: float = 12.0,
    yaw_gate_deg: float = 45.0,
    position_weight: float = 0.015,
    yaw_weight: float = 0.001,
) -> list[dict]:
    """用官方粗先验消除重复结构歧义，仍以点云几何决定精确位姿。"""

    output = []
    for row in rows:
        distance = math.hypot(float(row["x"]) - float(prior["x"]), float(row["y"]) - float(prior["y"]))
        yaw_error_deg = abs(
            math.degrees(wrap_angle(float(row["yaw"]) - float(prior["yaw"])))
        )
        if distance > float(position_gate_m) or yaw_error_deg > float(yaw_gate_deg):
            continue
        item = dict(row)
        geometry_score = float(row.get("geometry_score", row["score"]))
        penalty = (
            float(position_weight) * max(0.0, distance - float(position_deadband_m))
            + float(yaw_weight) * max(0.0, yaw_error_deg - float(yaw_deadband_deg))
        )
        item["geometry_score"] = geometry_score
        item["coarse_prior_distance_m"] = float(distance)
        item["coarse_prior_yaw_error_deg"] = float(yaw_error_deg)
        item["coarse_prior_penalty"] = float(penalty)
        item["score"] = geometry_score + float(penalty)
        output.append(item)
    return sorted(output, key=lambda item: float(item["score"]))


def coarse_prior_seeds(
    scorer: FrameScorer,
    template: np.ndarray,
    groups: dict,
    prior: dict,
    *,
    translation_step_m: float = 4.0,
    yaw_step_deg: float = 12.0,
) -> list[dict]:
    """在粗先验邻域放置稀疏种子，覆盖位置和朝向扰动。"""

    offsets = (-float(translation_step_m), 0.0, float(translation_step_m))
    yaw_offsets = tuple(math.radians(value) for value in (-float(yaw_step_deg), 0.0, float(yaw_step_deg)))
    rows = [
        scorer.score(
            template,
            groups,
            (
                float(prior["x"]) + dx,
                float(prior["y"]) + dy,
                float(prior["yaw"]) + dyaw,
            ),
        )
        for dx in offsets
        for dy in offsets
        for dyaw in yaw_offsets
    ]
    return _deduplicate_rows(rows, len(rows))


def coarse_correlation_seeds(scorer, template, groups, prior, *, radius=4., resolution=.15, limit=8):
    """在粗先验邻域连续搜索平移，避免仅用四米步长种子漏掉真实码头。"""
    if fftconvolve is None or not len(scorer.points):
        return coarse_prior_seeds(scorer, template, groups, prior)
    candidate_rows = []
    for offset in np.deg2rad(np.arange(-12., 12.1, 3.)):
        yaw = float(prior["yaw"]) + float(offset)
        model = rotate(template, yaw)
        kernel_cells = np.rint(model / resolution).astype(int)
        kernel_origin = kernel_cells.min(axis=0)
        kernel_cells -= kernel_origin
        kernel = np.zeros((kernel_cells[:, 1].max() + 1, kernel_cells[:, 0].max() + 1))
        kernel[kernel_cells[:, 1], kernel_cells[:, 0]] = 1.
        origin = np.floor((np.asarray([prior["x"], prior["y"]]) + kernel_origin * resolution - radius) / resolution) * resolution
        observed = np.zeros((kernel.shape[0] + int(math.ceil(2 * radius / resolution)),
                             kernel.shape[1] + int(math.ceil(2 * radius / resolution))))
        cells = np.rint((scorer.points - origin) / resolution).astype(int)
        valid = ((cells[:, 0] >= 0) & (cells[:, 0] < observed.shape[1])
                 & (cells[:, 1] >= 0) & (cells[:, 1] < observed.shape[0]))
        observed[cells[valid, 1], cells[valid, 0]] = 1.
        correlation = fftconvolve(observed, kernel[::-1, ::-1], mode="valid")
        keep = min(12, correlation.size)
        for index in np.argpartition(correlation.ravel(), -keep)[-keep:]:
            row, column = np.unravel_index(index, correlation.shape)
            translation = origin + np.asarray([column, row]) * resolution - kernel_origin * resolution
            candidate_rows.append((float(correlation[row, column]), *map(float, translation), yaw))
    peaks = nms(candidate_rows, limit=max(3, limit))
    seeds = [scorer.score(template, groups, (x, y, yaw)) for _, x, y, yaw in peaks]
    seeds.append(scorer.score(template, groups, (prior["x"], prior["y"], prior["yaw"])))
    return apply_coarse_pose_prior(seeds, prior)


def refine_hypothesis_with_coarse_prior(
    scorer: FrameScorer,
    template: np.ndarray,
    groups: dict,
    seed: dict,
    prior: dict,
    *,
    schedule=PRIOR_REFINE_SCHEDULE,
    max_iterations: int = 3,
    deadline: float | None = None,
) -> dict:
    """在粗先验约束下细化，避免优化过程跳到相邻泊位或背景立柱。"""

    current_rows = apply_coarse_pose_prior([seed], prior)
    if not current_rows:
        return dict(seed)
    current = current_rows[0]
    for step_m, step_yaw_deg in schedule:
        step_yaw = math.radians(float(step_yaw_deg))
        for _ in range(max_iterations):
            if deadline is not None and time.perf_counter() >= deadline:
                return current
            pose_rows = [
                scorer.score(template, groups, (current["x"], current["y"], current["yaw"])),
                scorer.score(template, groups, (current["x"] + step_m, current["y"], current["yaw"])),
                scorer.score(template, groups, (current["x"] - step_m, current["y"], current["yaw"])),
                scorer.score(template, groups, (current["x"], current["y"] + step_m, current["yaw"])),
                scorer.score(template, groups, (current["x"], current["y"] - step_m, current["yaw"])),
                scorer.score(template, groups, (current["x"], current["y"], current["yaw"] + step_yaw)),
                scorer.score(template, groups, (current["x"], current["y"], current["yaw"] - step_yaw)),
            ]
            ranked = apply_coarse_pose_prior(pose_rows, prior)
            if not ranked or float(ranked[0]["score"]) >= float(current["score"]) - 1e-9:
                break
            current = ranked[0]
    return current


def refine_icp_with_prior(scorer, template, groups, prior, *, maximum_iterations=24, deadline=None):
    """以许可的粗先验初始化鲁棒二维 ICP；测量几何决定精确位姿。"""
    pose = np.asarray([prior["x"], prior["y"], prior["yaw"]], dtype=float)
    best = scorer.score(template, groups, tuple(pose))
    if scorer.tree is None:
        return best
    for _ in range(maximum_iterations):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        transformed = rotate(template, pose[2]) + pose[:2]
        distances, indices = scorer.tree.query(transformed)
        valid = np.isfinite(distances) & (distances <= 2.)
        if np.count_nonzero(valid) < 20:
            break
        cutoff = np.quantile(distances[valid], .70)
        keep = valid & (distances <= max(.12, cutoff))
        source, target = template[keep], scorer.points[indices[keep]]
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        left, _, right = np.linalg.svd((source - source_mean).T @ (target - target_mean))
        rotation = right.T @ left.T
        if np.linalg.det(rotation) < 0.:
            right[-1] *= -1
            rotation = right.T @ left.T
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        translation = target_mean - rotation @ source_mean
        updated = np.asarray([*translation, yaw])
        rows = apply_coarse_pose_prior([scorer.score(template, groups, tuple(updated))], prior)
        if not rows:
            break
        if rows[0]["score"] < best["score"]:
            best = rows[0]
        delta = np.linalg.norm(updated[:2] - pose[:2]) + abs(wrap_angle(updated[2] - pose[2]))
        pose = updated
        if delta < .001:
            break
    return best


def register_with_coarse_prior(
    points: np.ndarray,
    template: np.ndarray,
    groups: dict,
    prior: dict,
    *,
    beam_width: int = 9,
    deadline: float | None = None,
) -> tuple[list[dict], str]:
    """以合法 dock 粗先验初始化，再用当前点云精配准。"""

    scorer = FrameScorer(points)
    seeds = coarse_correlation_seeds(scorer, template, groups, prior, limit=max(3, int(beam_width)))
    ranked = apply_coarse_pose_prior(seeds, prior)
    retained = _deduplicate_rows(ranked, max(int(beam_width), 3))
    refined = []
    for seed in retained:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        seed_prior = {key: seed[key] for key in ("x", "y", "yaw")}
        icp = refine_icp_with_prior(scorer, template, groups, seed_prior, deadline=deadline)
        refined.append(refine_hypothesis_with_coarse_prior(scorer, template, groups, icp, prior, deadline=deadline))
    refined = apply_coarse_pose_prior(refined, prior)
    refined = _deduplicate_rows(refined, max(int(beam_width), 3))
    if not refined:
        return refined, "NO_ESTIMATE"
    top = refined[0]
    competing_modes = [
        row
        for row in refined[1:]
        if math.hypot(float(row["x"]) - float(top["x"]), float(row["y"]) - float(top["y"])) > 2.0
        or abs(math.degrees(wrap_angle(float(row["yaw"]) - float(top["yaw"])))) > 8.0
    ]
    ambiguous = bool(
        competing_modes
        and float(competing_modes[0]["score"]) - float(top["score"]) < 0.02
    )
    return refined, "AMBIGUOUS" if ambiguous else "UNIQUE"


def _deduplicate_rows(rows: list[dict], limit: int) -> list[dict]:
    output = []
    for row in sorted(rows, key=lambda item: float(item["score"])):
        if any(
            math.hypot(float(row["x"]) - float(other["x"]), float(row["y"]) - float(other["y"])) <= 0.8
            and abs(math.degrees(wrap_angle(float(row["yaw"]) - float(other["yaw"])))) <= 2.0
            for other in output
        ):
            continue
        output.append(row)
        if len(output) >= limit:
            break
    return output


def refine_hypothesis(
    scorer: FrameScorer,
    template: np.ndarray,
    groups: dict,
    seed: dict,
    schedule=REFINE_SCHEDULE,
    max_iterations: int = 2,
) -> dict:
    """对每个 retained hypothesis 做固定粗到细的 SE(2) 迭代。"""

    current = dict(seed)
    for step_m, step_yaw_deg in schedule:
        step_yaw = math.radians(float(step_yaw_deg))
        for _ in range(max_iterations):
            candidates = [
                current,
                scorer.score(template, groups, (current["x"] + step_m, current["y"], current["yaw"])),
                scorer.score(template, groups, (current["x"] - step_m, current["y"], current["yaw"])),
                scorer.score(template, groups, (current["x"], current["y"] + step_m, current["yaw"])),
                scorer.score(template, groups, (current["x"], current["y"] - step_m, current["yaw"])),
                scorer.score(template, groups, (current["x"], current["y"], current["yaw"] + step_yaw)),
                scorer.score(template, groups, (current["x"], current["y"], current["yaw"] - step_yaw)),
            ]
            best = min(candidates, key=lambda item: float(item["score"]))
            if float(best["score"]) >= float(current["score"]) - 1e-9:
                break
            current = best
    return current


def refine_beam(
    scorer: FrameScorer,
    template: np.ndarray,
    groups: dict,
    rows: list[dict],
    *,
    schedule=REFINE_SCHEDULE,
    max_iterations: int = 2,
    beam_width: int = BEAM_WIDTH,
) -> list[dict]:
    refined = [
        refine_hypothesis(
            scorer,
            template,
            groups,
            row,
            schedule=schedule,
            max_iterations=max_iterations,
        )
        for row in rows[:beam_width]
    ]
    return _deduplicate_rows(refined, beam_width)


def register_with_scorer(
    scorer: FrameScorer,
    template: np.ndarray,
    groups: dict,
    *,
    beam_width: int = BEAM_WIDTH,
    candidate_pool: int = 16,
    prior: dict | None = None,
    prior_weight: float = 0.0,
) -> tuple[list[dict], str]:
    raw = nms(full_fft_candidates(scorer.points, template), limit=candidate_pool)
    scored = [scorer.score(template, groups, (item[1], item[2], item[3])) for item in raw]
    ranked = apply_temporal_prior(scored, prior, prior_weight)
    retained = _deduplicate_rows(ranked, beam_width)
    refined = refine_beam(scorer, template, groups, retained, beam_width=beam_width)
    refined = apply_temporal_prior(refined, prior, prior_weight)
    return refined, hypothesis_status(refined)


def register(points, template, groups):
    return register_with_scorer(FrameScorer(points), template, groups)
