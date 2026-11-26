"""五路有组织点云到保守局部占据栅格的无 ROS 内核。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
import yaml

from .grid_core import GridConfig, RayEvidence, build_grid, connected_components
from .terrain_evidence import terrain_evidence


def load_footprint_polygons(path: str | Path) -> tuple[np.ndarray, ...]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    polygons = tuple(
        np.asarray(item["polygon_xy"], dtype=np.float64)
        for item in document.get("components_polygon_xy", [])
    )
    if not polygons or any(polygon.ndim != 2 or polygon.shape[1] != 2 for polygon in polygons):
        raise ValueError("SELF_FOOTPRINT_INVALID")
    return polygons


def points_in_polygons(points_xy: np.ndarray, polygons: tuple[np.ndarray, ...]) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    inside = np.zeros(len(points), dtype=bool)
    x_values, y_values = points[:, 0], points[:, 1]
    for polygon in polygons:
        left, bottom = np.min(polygon, axis=0)
        right, top = np.max(polygon, axis=0)
        candidate = (
            (x_values >= left) & (x_values <= right)
            & (y_values >= bottom) & (y_values <= top)
        )
        if not candidate.any():
            continue
        indices = np.flatnonzero(candidate)
        local_x = x_values[indices]
        local_y = y_values[indices]
        polygon_inside = np.zeros(len(indices), dtype=bool)
        previous_x, previous_y = polygon[-1]
        for current_x, current_y in polygon:
            crosses = ((current_y > local_y) != (previous_y > local_y))
            denominator = previous_y - current_y
            safe = denominator if abs(float(denominator)) > 1e-12 else 1.0
            with np.errstate(invalid="ignore"):
                intersection_x = (
                    (previous_x - current_x) * (local_y - current_y) / safe + current_x
                )
            polygon_inside ^= crosses & (local_x < intersection_x)
            previous_x, previous_y = current_x, current_y
        inside[indices] |= polygon_inside
    return inside


def organized_structure_features(
    points_sensor: np.ndarray,
    *,
    width: int,
    height: int,
    sensor_rotation_to_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取垂直面法向、跨线束同距差和同距邻居数量。"""

    points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    count = len(points)
    if width <= 1 or height <= 2 or count != int(width) * int(height):
        return (
            np.full(count, np.nan, dtype=np.float64),
            np.full(count, np.inf, dtype=np.float64),
            np.zeros(count, dtype=np.uint8),
        )
    grid = points.reshape(height, width, 3)
    finite = np.isfinite(grid).all(axis=2)
    ranges = np.linalg.norm(grid, axis=2)
    vertical_delta = np.full((height, width), np.inf, dtype=np.float64)
    neighbor_count = np.zeros((height, width), dtype=np.uint8)
    for offset in (1, 2):
        upper, lower = slice(None, -offset), slice(offset, None)
        with np.errstate(invalid="ignore"):
            delta = np.abs(ranges[upper, :] - ranges[lower, :])
            close = finite[upper, :] & finite[lower, :] & (delta <= 0.35)
        current_upper = vertical_delta[upper, :]
        current_lower = vertical_delta[lower, :]
        valid = finite[upper, :] & finite[lower, :]
        current_upper[valid] = np.minimum(current_upper[valid], delta[valid])
        current_lower[valid] = np.minimum(current_lower[valid], delta[valid])
        neighbor_count[upper, :] += close
        neighbor_count[lower, :] += close

    left, right = np.roll(grid, 1, axis=1), np.roll(grid, -1, axis=1)
    horizontal_valid = finite & np.roll(finite, 1, axis=1) & np.roll(finite, -1, axis=1)
    vertical_valid = np.zeros_like(finite)
    vertical_valid[1:-1] = finite[:-2] & finite[1:-1] & finite[2:]
    with np.errstate(invalid="ignore"):
        horizontal_tangent = right - left
        vertical_tangent = np.zeros_like(grid)
        vertical_tangent[1:-1] = grid[2:] - grid[:-2]
        normals = np.cross(horizontal_tangent, vertical_tangent)
    normal_norm = np.linalg.norm(normals, axis=2)
    normal_valid = horizontal_valid & vertical_valid & (normal_norm > 1e-8)
    normals_unit = np.zeros_like(normals)
    normals_unit[normal_valid] = normals[normal_valid] / normal_norm[normal_valid, None]
    normals_base = normals_unit @ np.asarray(sensor_rotation_to_base, dtype=np.float64).T
    normal_up = np.full((height, width), np.nan, dtype=np.float64)
    normal_up[normal_valid] = np.abs(normals_base[:, :, 2][normal_valid])
    return normal_up.reshape(-1), vertical_delta.reshape(-1), neighbor_count.reshape(-1)


def _water_surface_mask(
    points_base: np.ndarray,
    finite: np.ndarray,
    *,
    width: int,
    height: int,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    """识别横摇/纵摇后在船体坐标系中呈倾斜面的水面回波。

    水面不是垂直障碍，但在有组织点云中会因为邻线束距离接近被误判成
    结构。仿真中的水面有波动，且同一行可能混入远处模型，因此同时使用
    竖直法向、低高度层、横向覆盖范围和鲁棒平面拟合，避免把普通船体或
    码头的局部面误删。
    """
    count = int(len(points_base))
    mask = np.zeros(count, dtype=bool)
    if width < 16 or height < 4 or count != int(width) * int(height):
        return mask, 0, 0.0

    points = np.asarray(points_base, dtype=np.float64)
    finite = np.asarray(finite, dtype=bool)
    if candidate_mask is None:
        candidate_mask = finite
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool) & finite

    # 船体坐标系中的水面通常落在低高度层。候选层以 0.1 m 步进滑动，
    # 让横摇、纵摇和波面起伏不会因为固定 z 阈值而漏检。
    low_layer = candidate_mask & (points[:, 2] >= -1.0) & (points[:, 2] <= 1.5)
    candidate_indices = np.flatnonzero(low_layer)
    if len(candidate_indices) < 400:
        return mask, 0, 0.0

    best: tuple[float, np.ndarray, np.ndarray, np.ndarray, float] | None = None
    for layer_center in np.arange(-0.8, 1.21, 0.1):
        layer_indices = candidate_indices[
            np.abs(points[candidate_indices, 2] - float(layer_center)) <= 0.30
        ]
        if len(layer_indices) < 400:
            continue
        layer_points = points[layer_indices]
        center = np.mean(layer_points, axis=0)
        centered = layer_points - center
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        normal = np.asarray(vh[-1], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal /= norm
        if normal[2] < 0.0:
            normal = -normal
        if normal[2] < 0.90:
            continue
        residuals = np.abs(centered @ normal)
        inliers = residuals <= 0.25
        if int(np.count_nonzero(inliers)) < 400:
            continue
        inlier_points = layer_points[inliers]
        span = np.ptp(inlier_points[:, :2], axis=0)
        # 普通船体/码头局部平面很难同时覆盖两个方向八米以上。
        if float(span[0]) < 8.0 or float(span[1]) < 8.0:
            continue
        rms = float(np.sqrt(np.mean(residuals[inliers] * residuals[inliers])))
        score = float(np.count_nonzero(inliers)) / max(0.05, rms)
        if best is None or score > best[0]:
            best = (score, center, normal, layer_indices[inliers], rms)

    if best is None:
        return mask, 0, 0.0

    _, center, normal, inlier_indices, rms = best
    # 用拟合平面重新评估所有候选点，保留水面波动带；平面外的垂直
    # 障碍不会因恰好落在同一高度层而被删除。
    candidate_points = points[candidate_indices]
    residuals = np.abs((candidate_points - center) @ normal)
    keep = (residuals <= 0.30) & (np.abs(candidate_points[:, 2] - center[2]) <= 0.45)
    mask[candidate_indices[keep]] = True
    rows = np.unique(np.flatnonzero(mask) // int(width))
    return mask, int(len(rows)), float(rms)


def extract_cloud_evidence(
    points_sensor: np.ndarray,
    *,
    width: int,
    height: int,
    transform: np.ndarray,
    footprint_polygons: tuple[np.ndarray, ...],
    cfg: GridConfig,
    angular_bin_deg: float = 0.5,
    navigation_z_m: float | None = None,
    gravity_rotation: np.ndarray | None = None,
    mean_water_elevation_m: float = 0.,
    self_geometry=None,
    output_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, list[RayEvidence], list[RayEvidence], dict]:
    points_sensor = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points_sensor).all(axis=1)
    points_base = np.full_like(points_sensor, np.nan, dtype=np.float64)
    points_base[finite] = points_sensor[finite] @ transform[:3, :3].T + transform[:3, 3]
    ranges = np.linalg.norm(points_sensor, axis=1)
    radius_xy = np.linalg.norm(points_base[:, :2], axis=1)
    normal_up, vertical_delta, neighbor_count = organized_structure_features(
        points_sensor,
        width=width,
        height=height,
        sensor_rotation_to_base=transform[:3, :3] if gravity_rotation is None else gravity_rotation @ transform[:3, :3],
    )
    self_body = points_in_polygons(points_base[:, :2], footprint_polygons) & (
        (points_base[:, 2] >= -1.0) & (points_base[:, 2] <= 3.0)
    )
    # 生产使用完整可见部件三维包络，发动机回波与浮筒上方的外部障碍须分开。
    if self_geometry is not None:
        self_body = self_geometry.contains(points_base)
    terrain_mask, terrain_boundary, terrain_stats = np.zeros(len(points_base), dtype=bool), np.empty((0, 2)), {}
    if navigation_z_m is not None and gravity_rotation is not None:
        elevation = points_base @ gravity_rotation[2, :] + navigation_z_m
        terrain_eligible = finite & ~self_body & (ranges >= cfg.min_range) & (radius_xy <= cfg.max_range)
        water_surface, terrain_mask, terrain_boundary, terrain_stats = terrain_evidence(
            points_base, elevation, normal_up, neighbor_count >= 1, terrain_eligible, mean_water_elevation_m)
        water_plane_rows, water_plane_rms = 0, 0.
    else:
        water_surface, water_plane_rows, water_plane_rms = _water_surface_mask(
            points_base, finite, width=width, height=height,
            candidate_mask=(~self_body & (ranges >= cfg.min_range) & (radius_xy <= cfg.max_range)),
        )
    eligible = (
        finite
        & ~self_body
        & ~water_surface
        & (ranges >= cfg.min_range)
        & (radius_xy <= cfg.max_range)
        & (points_base[:, 2] >= -0.6)
        & (points_base[:, 2] <= 3.0)
    )
    # 水面法向接近 +Z，不能因跨线束距离相近就标为障碍；法向无效的边缘点
    # 仅在至少两个邻线束高度一致且距离差很小时保守补回。
    vertical_surface = (
        (neighbor_count >= 1)
        & (np.isfinite(normal_up) & (normal_up <= 0.72))
    ) | (
        ~np.isfinite(normal_up)
        & (neighbor_count >= 2)
        & np.isfinite(vertical_delta)
        & (vertical_delta <= 0.15)
    )
    # 水面回波和其边缘伪结构主要落在 base_link 的低高度层；仿真中它们
    # 会形成近船体的几米假墙。静态避障只采纳高于水面明显距离的垂直面，
    # 动态目标仍由候选检测链路独立处理，避免用这一门限替代动态感知。
    structure = (eligible & vertical_surface & (points_base[:, 2] >= 0.75)) | terrain_mask
    # 分类和自身过滤仍在原观测坐标中完成；输出按该帧历史姿态补偿运动。
    motion = np.eye(4) if output_transform is None else np.asarray(output_transform)
    points_output = points_base@motion[:3, :3].T+motion[:3, 3]
    if len(terrain_boundary):
        if navigation_z_m is not None and gravity_rotation is not None:
            z = (mean_water_elevation_m-navigation_z_m-terrain_boundary@gravity_rotation[2, :2])/gravity_rotation[2, 2]
        else:
            z = np.zeros(len(terrain_boundary))
        boundary_output = np.column_stack((terrain_boundary, z))@motion[:3, :3].T+motion[:3, 3]
        terrain_boundary = boundary_output[:, :2]
    occupied_xy = np.concatenate((points_output[structure, :2], terrain_boundary))

    ray_eligible = (
        finite
        & ~self_body
        & ~water_surface
        & (ranges >= cfg.min_range)
        & (radius_xy <= cfg.max_range)
    )
    ray_indices = np.flatnonzero(ray_eligible)
    finite_rays: list[RayEvidence] = []
    free_only_rays: list[RayEvidence] = []
    origin = np.asarray(transform[:2, 3], dtype=np.float64)
    output_origin = (transform[:3, 3]@motion[:3, :3].T+motion[:3, 3])[:2]
    if len(ray_indices):
        vectors = points_base[ray_indices, :2] - origin
        ray_distances = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        bin_width = math.radians(float(angular_bin_deg))
        bins = np.floor((angles + math.pi) / bin_width).astype(np.int64)
        for bin_id in np.unique(bins):
            local_indices = np.flatnonzero(bins == bin_id)
            original_indices = ray_indices[local_indices]
            structural_positions = local_indices[structure[original_indices]]
            if len(structural_positions):
                chosen_position = structural_positions[int(np.argmin(ray_distances[structural_positions]))]
                chosen = int(ray_indices[chosen_position])
                finite_rays.append(
                    RayEvidence(tuple(output_origin), tuple(points_output[chosen, :2]), cfg.max_range, True)
                )
            else:
                chosen_position = local_indices[int(np.argmax(ray_distances[local_indices]))]
                chosen = int(ray_indices[chosen_position])
                free_only_rays.append(
                    RayEvidence(tuple(output_origin), tuple(points_output[chosen, :2]), cfg.max_range, False)
                )
    stats = {
        "raw_points": int(len(points_sensor)),
        "finite_points": int(np.count_nonzero(finite)),
        "self_points": int(np.count_nonzero(self_body & finite)),
        "water_plane_points": int(np.count_nonzero(water_surface)),
        "water_plane_rows": int(water_plane_rows),
        "water_plane_rms": float(water_plane_rms),
        "eligible_points": int(np.count_nonzero(eligible)),
        "structure_points": int(np.count_nonzero(structure)),
        "finite_rays": len(finite_rays),
        "free_only_rays": len(free_only_rays),
        "terrain": terrain_stats,
    }
    return occupied_xy, finite_rays, free_only_rays, stats


def self_mask_grid(cfg: GridConfig, polygons: tuple[np.ndarray, ...]) -> np.ndarray:
    columns = cfg.origin_xy[0] + (np.arange(cfg.width) + 0.5) * cfg.resolution
    rows = cfg.origin_xy[1] + (np.arange(cfg.height) + 0.5) * cfg.resolution
    xx, yy = np.meshgrid(columns, rows)
    return points_in_polygons(np.column_stack((xx.reshape(-1), yy.reshape(-1))), polygons).reshape(
        cfg.height, cfg.width
    )


def inflate_occupied(grid: np.ndarray, cfg: GridConfig, radius_m: float) -> np.ndarray:
    """膨胀只标记未知区的缓冲代价50，不制造实物回波或覆盖清空证据。"""
    radius_cells = max(0, int(math.ceil(float(radius_m) / cfg.resolution)))
    if radius_cells <= 0:
        return np.asarray(grid, dtype=np.int8).copy()
    axis = np.arange(-radius_cells, radius_cells + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = (xx * xx + yy * yy) <= (float(radius_m) / cfg.resolution) ** 2
    occupied = binary_dilation(np.asarray(grid) == 100, structure=kernel)
    output = np.asarray(grid, dtype=np.int8).copy()
    # 100为实际占用，0为射线证实空闲；规划已按全船体和格子尺寸验碰撞。
    # 两者不能被膨胀改成另一批实物，否则窄泊位会额外缩小且旧占用无法清除。
    output[occupied & (output == -1)] = 50
    return output


def build_environment_grid(
    occupied_parts: list[np.ndarray],
    finite_rays: list[RayEvidence],
    free_only_rays: list[RayEvidence],
    *,
    cfg: GridConfig,
    self_mask: np.ndarray,
    inflation_radius_m: float,
    minimum_occupied_cell_hits: int = 2,
) -> tuple[np.ndarray, list[dict], dict]:
    combined = np.concatenate(
        [np.asarray(part, dtype=np.float64).reshape(-1, 2) for part in occupied_parts if len(part)],
        axis=0,
    ) if any(len(part) for part in occupied_parts) else np.empty((0, 2), dtype=np.float64)
    if len(combined):
        cells = np.floor((combined - np.asarray(cfg.origin_xy)) / cfg.resolution).astype(np.int64)
        valid = (
            (cells[:, 0] >= 0) & (cells[:, 0] < cfg.width)
            & (cells[:, 1] >= 0) & (cells[:, 1] < cfg.height)
        )
        unique_cells, counts = np.unique(cells[valid], axis=0, return_counts=True)
        retained = unique_cells[counts >= max(1, int(minimum_occupied_cell_hits))]
        occupied_xy = np.column_stack(
            (
                cfg.origin_xy[0] + (retained[:, 0] + 0.5) * cfg.resolution,
                cfg.origin_xy[1] + (retained[:, 1] + 0.5) * cfg.resolution,
            )
        ) if len(retained) else np.empty((0, 2), dtype=np.float64)
    else:
        unique_cells = np.empty((0, 2), dtype=np.int64)
        occupied_xy = np.empty((0, 2), dtype=np.float64)
    # 点云体素是唯一占据证据；所有射线只用于清空到首个已冻结占据格，避免
    # 单个边缘回波绕过体素支持门重新写成障碍。
    carving_rays = [
        RayEvidence(ray.origin_xy, ray.endpoint_xy, ray.max_range_m, False, ray.range_clipped)
        for ray in finite_rays + free_only_rays
    ]
    raw_grid, stats = build_grid(
        [occupied_xy],
        [],
        carving_rays,
        cfg,
        self_mask=self_mask,
    )
    compact, grid_only = connected_components(raw_grid, cfg, min_cells=2)
    inflated = inflate_occupied(raw_grid, cfg, inflation_radius_m)
    inflated[self_mask] = -1
    stats = dict(stats)
    stats["raw_occupied_cells"] = int(np.count_nonzero(raw_grid == 100))
    stats["candidate_occupied_cells"] = int(len(unique_cells))
    stats["minimum_occupied_cell_hits"] = int(minimum_occupied_cell_hits)
    stats["inflated_occupied_cells"] = int(np.count_nonzero(inflated >= 50))
    stats["padding_cells"] = int(np.count_nonzero(inflated == 50))
    stats["compact_components"] = compact
    stats["grid_only_components"] = grid_only
    return inflated, compact, stats


__all__ = [
    "build_environment_grid",
    "extract_cloud_evidence",
    "inflate_occupied",
    "load_footprint_polygons",
    "organized_structure_features",
    "points_in_polygons",
    "self_mask_grid",
]
