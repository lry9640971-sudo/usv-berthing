#!/usr/bin/env python3
"""Phase16S 的确定性 OccupancyGrid 核心。

实现先合并全部 OCCUPIED 证据，再做保守 FREE carving。所有计算只接收
传感器坐标变换后的数据，不读取真值或世界话题。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import label, find_objects


@dataclass(frozen=True)
class GridConfig:
    resolution: float = 0.20
    width: int = 400
    height: int = 400
    origin_xy: tuple[float, float] = (-40.0, -40.0)
    min_range: float = 0.5
    max_range: float = 40.0
    variant: str = "V0_OCCUPIED_FIRST"


@dataclass(frozen=True)
class RayEvidence:
    origin_xy: tuple[float, float]
    endpoint_xy: tuple[float, float] | None
    max_range_m: float
    finite: bool
    range_clipped: bool = False


def _cell(xy: tuple[float, float] | np.ndarray, cfg: GridConfig) -> tuple[int, int] | None:
    """将世界 xy 转为 (row, column)，边界外返回 None。"""
    x, y = float(xy[0]), float(xy[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    col = int(math.floor((x - cfg.origin_xy[0]) / cfg.resolution))
    row = int(math.floor((y - cfg.origin_xy[1]) / cfg.resolution))
    return (row, col) if 0 <= row < cfg.height and 0 <= col < cfg.width else None


def _clip_segment_to_grid(start: np.ndarray, end: np.ndarray, cfg: GridConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang--Barsky 裁剪，支持起点在格内、终点在格外。"""
    xmin, ymin = cfg.origin_xy
    xmax, ymax = xmin + cfg.width * cfg.resolution, ymin + cfg.height * cfg.resolution
    d = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], start[0] - xmin), (d[0], xmax - start[0]), (-d[1], start[1] - ymin), (d[1], ymax - start[1])):
        if abs(float(p)) < 1e-15:
            if float(q) < 0.0:
                return None
            continue
        t = float(q) / float(p)
        if p < 0.0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return np.asarray(start, dtype=np.float64) + t0 * d, np.asarray(start, dtype=np.float64) + t1 * d


def _bresenham(start: tuple[int, int], end: tuple[int, int], cfg: GridConfig) -> list[tuple[int, int]]:
    r0, c0 = int(start[0]), int(start[1]); r1, c1 = int(end[0]), int(end[1])
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1
    err = dc - dr; out: list[tuple[int, int]] = []
    while True:
        if 0 <= r0 < cfg.height and 0 <= c0 < cfg.width:
            out.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        twice = 2 * err
        if twice > -dr:
            err -= dr; c0 += sc
        if twice < dc:
            err += dr * 0 + dc; r0 += sr
        if len(out) > (cfg.width + cfg.height) * 4:
            break
    return out


def _ray_cells(ray: RayEvidence, cfg: GridConfig) -> tuple[list[tuple[int, int]], bool]:
    if ray.endpoint_xy is None:
        return [], False
    start = np.asarray(ray.origin_xy, dtype=np.float64); end = np.asarray(ray.endpoint_xy, dtype=np.float64)
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        return [], False
    clipped = _clip_segment_to_grid(start, end, cfg)
    if clipped is None:
        return [], False
    start_clip, end_clip = clipped
    eps = cfg.resolution * 1e-9
    if _cell(start_clip, cfg) is None:
        direction = end_clip - start_clip; n = float(np.linalg.norm(direction))
        if n <= 1e-15: return [], False
        start_clip = start_clip + direction / n * eps
    if _cell(end_clip, cfg) is None:
        direction = start_clip - end_clip; n = float(np.linalg.norm(direction))
        if n <= 1e-15: return [], False
        end_clip = end_clip + direction / n * eps
    start_cell = _cell(start_clip, cfg); end_cell = _cell(end_clip, cfg)
    if start_cell is None or end_cell is None:
        return [], False
    return _bresenham(start_cell, end_cell, cfg), _cell(end, cfg) is not None


def _finite_xy(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)
    values = values[:, :2]
    return values[np.isfinite(values).all(axis=1)]


def connected_components(grid: np.ndarray, cfg: GridConfig, min_cells: int = 2,
                         max_area_m2: float = 20.0, max_extent_m: float = 8.0) -> tuple[list[dict], list[dict]]:
    """紧凑占据域输出真实质心；连续大区域只保留在 Grid。"""
    labels, count = label(np.asarray(grid) == 100, np.ones((3, 3), dtype=bool))
    compact, grid_only = [], []
    if not count:
        return compact, grid_only
    rows, columns = np.indices(labels.shape)
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    sums_x = np.bincount(labels.ravel(), weights=columns.ravel(), minlength=count + 1)
    sums_y = np.bincount(labels.ravel(), weights=rows.ravel(), minlength=count + 1)
    for index, slices in enumerate(find_objects(labels), start=1):
        if slices is None:
            continue
        ys, xs = slices
        size = int(counts[index])
        item = {"cell_count": size, "area_m2": float(size * cfg.resolution ** 2),
                "bbox_min_xy_m": [cfg.origin_xy[0] + (xs.start + .5) * cfg.resolution,
                                  cfg.origin_xy[1] + (ys.start + .5) * cfg.resolution],
                "bbox_max_xy_m": [cfg.origin_xy[0] + (xs.stop - .5) * cfg.resolution,
                                  cfg.origin_xy[1] + (ys.stop - .5) * cfg.resolution],
                "centroid_xy_m": [cfg.origin_xy[0] + (sums_x[index] / size + .5) * cfg.resolution,
                                  cfg.origin_xy[1] + (sums_y[index] / size + .5) * cfg.resolution]}
        extent = max(xs.stop - xs.start, ys.stop - ys.start) * cfg.resolution
        if size >= min_cells and item["area_m2"] <= max_area_m2 and extent <= max_extent_m:
            compact.append(item)
        else:
            item["rejection_reason"] = "continuous_large_region_or_below_min_cells"; grid_only.append(item)
    return compact, grid_only


def _normalised_ray(ray: RayEvidence, cfg: GridConfig) -> RayEvidence | None:
    origin = np.asarray(ray.origin_xy, dtype=np.float64)
    endpoint = None if ray.endpoint_xy is None else np.asarray(ray.endpoint_xy, dtype=np.float64)
    if endpoint is None or not np.isfinite(origin).all() or not np.isfinite(endpoint).all():
        return None
    distance = float(np.linalg.norm(endpoint - origin)); limit = min(float(cfg.max_range), float(ray.max_range_m))
    if not math.isfinite(distance) or not math.isfinite(limit) or limit < cfg.min_range or distance < cfg.min_range:
        return None
    if distance > limit:
        direction = (endpoint - origin) / distance; endpoint = origin + direction * limit
        return RayEvidence(tuple(origin), tuple(endpoint), limit, bool(ray.finite), True)
    return RayEvidence(tuple(origin), tuple(endpoint), limit, bool(ray.finite))


def build_grid(point_clouds_xy: Iterable[np.ndarray], finite_rays: Iterable[RayEvidence], inf_rays: Iterable[RayEvidence],
               cfg: GridConfig = GridConfig(), self_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """冻结所有占据证据后进行 FREE carving，保证传感器顺序不影响结果。"""
    if cfg.variant not in {"V0_OCCUPIED_FIRST", "V1_NO_INF_FREE"}:
        raise ValueError(f"unsupported variant: {cfg.variant}")
    mask = None if self_mask is None else np.asarray(self_mask, dtype=bool)
    if mask is not None and mask.shape != (cfg.height, cfg.width): raise ValueError("self_mask shape mismatch")
    occupied = np.zeros((cfg.height, cfg.width), dtype=bool)
    occupied_from_points = 0; occupied_from_scan = 0; skipped_self_points = 0; invalid_points = 0
    point_cells: list[tuple[int, int]] = []
    for values in point_clouds_xy:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0: continue
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        finite = _finite_xy(arr); invalid_points += int(arr.shape[0] - finite.shape[0])
        for xy in finite:
            cell = _cell(xy, cfg)
            if cell is not None: point_cells.append(cell)
    for cell in sorted(set(point_cells)):
        if mask is not None and mask[cell]: skipped_self_points += 1; continue
        if not occupied[cell]: occupied_from_points += 1
        occupied[cell] = True
    # finite_rays 中超过 max_range 的测量仍需清空到限制距离，因此保留其 finite 来源标记和 range_clipped。
    finite_norm = [v for ray in finite_rays if (v := _normalised_ray(ray, cfg)) is not None]
    inf_norm = [v for ray in inf_rays if (v := _normalised_ray(ray, cfg)) is not None and not v.finite]
    for ray in sorted(finite_norm, key=lambda v: (v.origin_xy, v.endpoint_xy or (0.0, 0.0))):
        endpoint_cell = _cell(ray.endpoint_xy, cfg)
        if endpoint_cell is None: continue
        if mask is not None and mask[endpoint_cell]: continue
        if ray.finite and not ray.range_clipped:
            if not occupied[endpoint_cell]: occupied_from_scan += 1
            occupied[endpoint_cell] = True
    grid = np.full((cfg.height, cfg.width), -1, dtype=np.int8)
    free_cells = 0; rays_stopped_by_occupied = 0; rays_stopped_by_self = 0; clipped_rays = 0
    rays = finite_norm + (inf_norm if cfg.variant == "V0_OCCUPIED_FIRST" else [])
    for ray in sorted(rays, key=lambda v: (v.origin_xy, v.endpoint_xy or (0.0, 0.0), v.finite)):
        cells, endpoint_inside = _ray_cells(ray, cfg)
        if not cells: continue
        if not endpoint_inside: clipped_rays += 1
        left_self = False
        for index, cell in enumerate(cells):
            is_endpoint = index == len(cells) - 1
            if mask is not None and bool(mask[cell]):
                if not left_self: continue
                rays_stopped_by_self += 1; break
            left_self = True
            if occupied[cell]:
                rays_stopped_by_occupied += int(not (is_endpoint and ray.finite and endpoint_inside)); break
            if is_endpoint and ray.finite and endpoint_inside and not ray.range_clipped: break
            if grid[cell] != 0:
                grid[cell] = 0; free_cells += 1
    grid[occupied] = 100
    if mask is not None: grid[mask & ~occupied] = -1
    compact, grid_only = connected_components(grid, cfg)
    stats = {
        "occupied_from_points_cells": int(occupied_from_points), "occupied_from_scan_cells": int(occupied_from_scan),
        "occupied_cells": int(np.count_nonzero(grid == 100)), "free_cells": int(np.count_nonzero(grid == 0)),
        "unknown_cells": int(np.count_nonzero(grid == -1)), "total_cells": int(grid.size), "free_carving_new_cells": int(free_cells),
        "rays_stopped_by_occupied": int(rays_stopped_by_occupied), "rays_stopped_by_self": int(rays_stopped_by_self),
        "clipped_ray_count": int(clipped_rays), "skipped_self_points": int(skipped_self_points), "invalid_point_count": int(invalid_points),
        "finite_ray_count": int(len(finite_norm)), "inf_ray_count": int(len(inf_norm)),
        "inf_free_applied": bool(cfg.variant == "V0_OCCUPIED_FIRST"), "compact_components": compact, "grid_only_components": grid_only,
    }
    return grid, stats


def cloud_xyz(message: Any, point_cloud2: Any) -> np.ndarray:
    """从 ROS PointCloud2 提取 x/y/z；失败时返回空数组并由上层计数。"""
    try:
        values = point_cloud2.read_points_numpy(message, field_names=["x", "y", "z"], skip_nans=False)
    except Exception:
        try: values = list(point_cloud2.read_points(message, field_names=["x", "y", "z"], skip_nans=False))
        except Exception: return np.empty((0, 3), dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1, 3)


__all__ = ["GridConfig", "RayEvidence", "build_grid", "connected_components", "cloud_xyz", "_cell", "_ray_cells"]
