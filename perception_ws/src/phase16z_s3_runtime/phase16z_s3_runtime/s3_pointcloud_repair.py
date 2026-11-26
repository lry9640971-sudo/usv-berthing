"""PointCloud2 三维候选提取。

这里不读取真值、Gazebo 或控制话题。输入点保留在 sensor 帧，经已证明固定的
sensor->base 刚体外参转换后，在三维空间做本船过滤、体素降采样和连通聚类。
候选公共消息仍沿用现有的 XY 接口，但 receipt 会保存三维统计，便于下一阶段
检查 Z 范围、点源和目标身份。
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .s3_candidate_repair import CandidateObservation, FootprintUnion
from .s3_timestamped_frames import RigidTransform3D


@dataclass(frozen=True)
class BasePoint3D:
    stamp_ns: int
    source_sensor: str
    point_index: int
    x_m: float
    y_m: float
    z_m: float
    range_m: float
    organized_normal_up_abs: float = math.nan
    min_vertical_range_delta_m: float = math.inf
    vertical_consistent_neighbor_count: int = 0


def organized_point_features(
    points_xyz_m: np.ndarray,
    *,
    width: int,
    height: int,
    sensor_to_base: RigidTransform3D,
) -> tuple[np.ndarray, np.ndarray]:
    """从有组织点云提取局部法向和垂直通道同距证据。"""

    array = np.asarray(points_xyz_m, dtype=np.float64)
    if width <= 1 or height <= 2 or array.shape[0] != width * height:
        return np.full(array.shape[0], np.nan), np.full(array.shape[0], np.inf)
    grid = array[:, :3].reshape(height, width, 3)
    finite = np.isfinite(grid).all(axis=2)
    radial = np.hypot(grid[:, :, 0], grid[:, :, 1])
    vertical_delta = np.full((height, width), np.inf, dtype=np.float64)
    for offset in (1, 2):
        for source, target in ((slice(offset, None), slice(None, -offset)),
                               (slice(None, -offset), slice(offset, None))):
            valid = finite[source, :] & finite[target, :]
            with np.errstate(invalid="ignore"):
                delta = np.abs(radial[source, :] - radial[target, :])
            current = vertical_delta[target, :]
            current[valid] = np.minimum(current[valid], delta[valid])

    left = np.roll(grid, 1, axis=1)
    right = np.roll(grid, -1, axis=1)
    horizontal_valid = finite & np.roll(finite, 1, axis=1) & np.roll(finite, -1, axis=1)
    vertical_valid = np.zeros_like(finite)
    vertical_valid[1:-1] = finite[:-2] & finite[1:-1] & finite[2:]
    with np.errstate(invalid="ignore"):
        tangent_horizontal = right - left
        tangent_vertical = np.zeros_like(grid)
        tangent_vertical[1:-1] = grid[2:] - grid[:-2]
        normals_sensor = np.cross(tangent_horizontal, tangent_vertical)
    normal_norm = np.linalg.norm(normals_sensor, axis=2)
    normal_valid = horizontal_valid & vertical_valid & (normal_norm > 1e-8)
    normals_unit = np.zeros_like(normals_sensor)
    normals_unit[normal_valid] = normals_sensor[normal_valid] / normal_norm[normal_valid, None]
    quaternion = np.asarray((
        sensor_to_base.quaternion_x, sensor_to_base.quaternion_y,
        sensor_to_base.quaternion_z, sensor_to_base.quaternion_w,
    ), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    qx, qy, qz, qw = quaternion
    rotation = np.asarray((
        (1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)),
        (2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)),
        (2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)),
    ))
    normals_base = normals_unit @ rotation.T
    normal_up = np.full((height, width), np.nan, dtype=np.float64)
    normal_up[normal_valid] = np.abs(normals_base[:, :, 2][normal_valid])
    return normal_up.reshape(-1), vertical_delta.reshape(-1)


def organized_vertical_neighbor_counts(
    points_xyz_m: np.ndarray,
    *,
    width: int,
    height: int,
    maximum_delta_m: float = 0.30,
) -> np.ndarray:
    """统计每个回波在相邻四条垂直线束中的同距邻居数。"""

    array = np.asarray(points_xyz_m, dtype=np.float64)
    if width <= 1 or height <= 2 or array.shape[0] != width * height:
        return np.zeros(array.shape[0], dtype=np.uint8)
    grid = array[:, :3].reshape(height, width, 3)
    finite = np.isfinite(grid).all(axis=2)
    radial = np.hypot(grid[:, :, 0], grid[:, :, 1])
    counts = np.zeros((height, width), dtype=np.uint8)
    for offset in (1, 2):
        upper = slice(None, -offset)
        lower = slice(offset, None)
        with np.errstate(invalid="ignore"):
            close = (
                finite[upper, :]
                & finite[lower, :]
                & (np.abs(radial[upper, :] - radial[lower, :]) <= float(maximum_delta_m))
            )
        counts[upper, :] += close
        counts[lower, :] += close
    return counts.reshape(-1)


@dataclass
class PointCloudFunnel:
    input_points: int = 0
    finite_points: int = 0
    accepted_points: int = 0
    dropped_nonfinite: int = 0
    dropped_self_body: int = 0
    dropped_height: int = 0
    voxel_cells: int = 0
    clusters_seen: int = 0
    clusters_rejected_support: int = 0
    candidates_emitted: int = 0
    estimated_water_plane_frames: int = 0
    estimated_water_plane_projection_sum_m: float = 0.0
    estimated_water_plane_projection_min_m: float = math.inf
    estimated_water_plane_projection_max_m: float = -math.inf
    drop_reasons: Counter[str] = field(default_factory=Counter)

    def merge(self, other: "PointCloudFunnel") -> None:
        for name in (
            "input_points", "finite_points", "accepted_points", "dropped_nonfinite",
            "dropped_self_body", "dropped_height", "voxel_cells", "clusters_seen",
            "clusters_rejected_support", "candidates_emitted",
            "estimated_water_plane_frames",
        ):
            setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))
        self.estimated_water_plane_projection_sum_m += other.estimated_water_plane_projection_sum_m
        if other.estimated_water_plane_frames:
            self.estimated_water_plane_projection_min_m = min(
                self.estimated_water_plane_projection_min_m,
                other.estimated_water_plane_projection_min_m,
            )
            self.estimated_water_plane_projection_max_m = max(
                self.estimated_water_plane_projection_max_m,
                other.estimated_water_plane_projection_max_m,
            )
        self.drop_reasons.update(other.drop_reasons)

    def as_dict(self) -> dict[str, object]:
        return {
            "input_points": int(self.input_points),
            "finite_points": int(self.finite_points),
            "accepted_points": int(self.accepted_points),
            "dropped_nonfinite": int(self.dropped_nonfinite),
            "dropped_self_body": int(self.dropped_self_body),
            "dropped_height": int(self.dropped_height),
            "voxel_cells": int(self.voxel_cells),
            "clusters_seen": int(self.clusters_seen),
            "clusters_rejected_support": int(self.clusters_rejected_support),
            "candidates_emitted": int(self.candidates_emitted),
            "estimated_water_plane": {
                "frames": int(self.estimated_water_plane_frames),
                "mean_projection_m": (
                    self.estimated_water_plane_projection_sum_m / self.estimated_water_plane_frames
                    if self.estimated_water_plane_frames else None
                ),
                "minimum_projection_m": (
                    self.estimated_water_plane_projection_min_m
                    if self.estimated_water_plane_frames else None
                ),
                "maximum_projection_m": (
                    self.estimated_water_plane_projection_max_m
                    if self.estimated_water_plane_frames else None
                ),
            },
            "drop_reasons": dict(sorted(self.drop_reasons.items())),
        }


def pointcloud_to_base_points(
    *,
    stamp_ns: int,
    source_sensor: str,
    points_xyz_m: Iterable[Sequence[float]],
    sensor_to_base: RigidTransform3D,
    self_footprint: FootprintUnion,
    ownship_z_min_m: float = -1.0,
    ownship_z_max_m: float = 3.0,
    detection_z_min_m: float = -0.5,
    detection_z_max_m: float = 1.5,
    maximum_detection_range_m: float = 80.0,
    organized_normal_up_abs: np.ndarray | None = None,
    min_vertical_range_delta_m: np.ndarray | None = None,
    vertical_consistent_neighbor_count: np.ndarray | None = None,
    gravity_up_base: tuple[float, float, float] | None = None,
    height_origin_m: float = 0.0,
    estimate_height_origin_from_cloud: bool = False,
    funnel: PointCloudFunnel | None = None,
    self_geometry=None,
) -> tuple[BasePoint3D, ...]:
    """把 PointCloud2 的 XYZ 点转到 base，并执行保守的三维本船过滤。"""

    if int(stamp_ns) <= 0:
        raise ValueError("POINTCLOUD_STAMP_INVALID")
    if not source_sensor:
        raise ValueError("POINTCLOUD_SENSOR_NAME_MISSING")
    if self_geometry is not None and not isinstance(points_xyz_m, np.ndarray):
        points_xyz_m = np.asarray(list(points_xyz_m), dtype=float).reshape(-1, 3)
    if isinstance(points_xyz_m, np.ndarray):
        array = np.asarray(points_xyz_m, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError("POINTCLOUD_XYZ_SHAPE_INVALID")
        finite = np.isfinite(array[:, :3]).all(axis=1)
        finite_indices = np.flatnonzero(finite)
        values = array[finite, :3]
        if funnel is not None:
            funnel.input_points += int(array.shape[0])
            funnel.finite_points += int(values.shape[0])
            skipped = int(array.shape[0] - values.shape[0])
            funnel.dropped_nonfinite += skipped
            if skipped:
                funnel.drop_reasons["nonfinite_xyz"] += skipped
        if values.size == 0:
            return ()
        norm = math.sqrt(sum(float(value) ** 2 for value in (
            sensor_to_base.quaternion_x,
            sensor_to_base.quaternion_y,
            sensor_to_base.quaternion_z,
            sensor_to_base.quaternion_w,
        )))
        qx, qy, qz, qw = (
            float(sensor_to_base.quaternion_x) / norm,
            float(sensor_to_base.quaternion_y) / norm,
            float(sensor_to_base.quaternion_z) / norm,
            float(sensor_to_base.quaternion_w) / norm,
        )
        px, py, pz = values[:, 0], values[:, 1], values[:, 2]
        x_base = (1.0 - 2.0 * (qy * qy + qz * qz)) * px + 2.0 * (qx * qy - qz * qw) * py + 2.0 * (qx * qz + qy * qw) * pz + float(sensor_to_base.translation_x_m)
        y_base = 2.0 * (qx * qy + qz * qw) * px + (1.0 - 2.0 * (qx * qx + qz * qz)) * py + 2.0 * (qy * qz - qx * qw) * pz + float(sensor_to_base.translation_y_m)
        z_base = 2.0 * (qx * qz - qy * qw) * px + 2.0 * (qy * qz + qx * qw) * py + (1.0 - 2.0 * (qx * qx + qy * qy)) * pz + float(sensor_to_base.translation_z_m)
        if self_geometry is not None:
            ownship = self_geometry.contains(np.column_stack((x_base, y_base, z_base)))
        else:
            inside = np.zeros(values.shape[0], dtype=bool)
            for polygon, bounds in zip(self_footprint.components, self_footprint.component_bounds):
                left, right, bottom, top = bounds
                candidate = (x_base >= left) & (x_base <= right) & (y_base >= bottom) & (y_base <= top)
                if not candidate.any():
                    continue
                polygon_inside = np.zeros(values.shape[0], dtype=bool)
                previous_x, previous_y = polygon[-1]
                for current_x, current_y in polygon:
                    crosses = ((current_y > y_base) != (previous_y > y_base))
                    denominator = previous_y - current_y
                    safe_denominator = denominator if abs(denominator) > 1e-12 else 1.0
                    intersection_x = (previous_x - current_x) * (y_base - current_y) / safe_denominator + current_x
                    polygon_inside ^= crosses & (x_base < intersection_x)
                    previous_x, previous_y = current_x, current_y
                inside |= candidate & polygon_inside
            ownship = inside & (z_base >= float(ownship_z_min_m)) & (z_base <= float(ownship_z_max_m))
        if funnel is not None:
            dropped = int(ownship.sum())
            funnel.dropped_self_body += dropped
            if dropped:
                funnel.drop_reasons["true_ownship_footprint_3d"] += dropped
        sensor_ranges = np.sqrt(np.sum(values * values, axis=1))
        if gravity_up_base is None:
            height = z_base
        else:
            projected_height = (
                gravity_up_base[0] * x_base
                + gravity_up_base[1] * y_base
                + gravity_up_base[2] * z_base
            )
            if estimate_height_origin_from_cloud:
                eligible = ~ownship & (sensor_ranges >= 3.0) & (sensor_ranges <= float(maximum_detection_range_m))
                values = projected_height[eligible]
                values = values[np.isfinite(values) & (values >= -4.0) & (values <= 4.0)]
                if values.size >= 100:
                    bin_edges = np.arange(-4.0, 4.0201, 0.02)
                    histogram, _ = np.histogram(values, bins=bin_edges)
                    peak_index = int(np.argmax(histogram))
                    lower, upper = bin_edges[peak_index], bin_edges[peak_index + 1]
                    peak_values = values[(values >= lower) & (values < upper)]
                    water_projection = float(np.median(peak_values))
                    height_origin_m = -water_projection
                    if funnel is not None:
                        funnel.estimated_water_plane_frames += 1
                        funnel.estimated_water_plane_projection_sum_m += water_projection
                        funnel.estimated_water_plane_projection_min_m = min(
                            funnel.estimated_water_plane_projection_min_m, water_projection
                        )
                        funnel.estimated_water_plane_projection_max_m = max(
                            funnel.estimated_water_plane_projection_max_m, water_projection
                        )
            height = projected_height + height_origin_m
        candidate_height = (height >= float(detection_z_min_m)) & (height <= float(detection_z_max_m))
        candidate_range = sensor_ranges <= float(maximum_detection_range_m)
        if funnel is not None:
            dropped_height = int((~ownship & ~candidate_height).sum())
            funnel.dropped_height += dropped_height
            if dropped_height:
                funnel.drop_reasons["candidate_height_gate"] += dropped_height
            dropped_range = int((~ownship & candidate_height & ~candidate_range).sum())
            if dropped_range:
                funnel.drop_reasons["candidate_range_gate"] += dropped_range
        keep = ~ownship & candidate_height & candidate_range
        accepted_indices = finite_indices[keep]
        accepted_xyz = np.column_stack((x_base[keep], y_base[keep], z_base[keep]))
        ranges = sensor_ranges[keep]
        if funnel is not None:
            funnel.accepted_points += int(accepted_xyz.shape[0])
        return tuple(
            BasePoint3D(
                int(stamp_ns),
                str(source_sensor),
                int(point_index),
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                float(distance),
                organized_normal_up_abs=float(organized_normal_up_abs[point_index]) if organized_normal_up_abs is not None else math.nan,
                min_vertical_range_delta_m=float(min_vertical_range_delta_m[point_index]) if min_vertical_range_delta_m is not None else math.inf,
                vertical_consistent_neighbor_count=int(vertical_consistent_neighbor_count[point_index]) if vertical_consistent_neighbor_count is not None else 0,
            )
            for point_index, xyz, distance in zip(accepted_indices, accepted_xyz, ranges)
        )

    result: list[BasePoint3D] = []
    for point_index, point in enumerate(points_xyz_m):
        if funnel is not None:
            funnel.input_points += 1
        if len(point) < 3:
            if funnel is not None:
                funnel.drop_reasons["point_xyz_missing"] += 1
            continue
        x_sensor, y_sensor, z_sensor = (float(point[index]) for index in range(3))
        if not all(math.isfinite(value) for value in (x_sensor, y_sensor, z_sensor)):
            if funnel is not None:
                funnel.dropped_nonfinite += 1
                funnel.drop_reasons["nonfinite_xyz"] += 1
            continue
        if funnel is not None:
            funnel.finite_points += 1
        x_base, y_base, z_base = sensor_to_base.rotate_translate((x_sensor, y_sensor, z_sensor))
        if not all(math.isfinite(value) for value in (x_base, y_base, z_base)):
            if funnel is not None:
                funnel.drop_reasons["transformed_xyz_nonfinite"] += 1
            continue
        if self_footprint.contains(x_base, y_base) and ownship_z_min_m <= z_base <= ownship_z_max_m:
            if funnel is not None:
                funnel.dropped_self_body += 1
                funnel.drop_reasons["true_ownship_footprint_3d"] += 1
            continue
        distance = math.sqrt(x_sensor * x_sensor + y_sensor * y_sensor + z_sensor * z_sensor)
        height = z_base if gravity_up_base is None else sum(a*b for a,b in zip(gravity_up_base,(x_base,y_base,z_base))) + height_origin_m
        if not (detection_z_min_m <= height <= detection_z_max_m):
            if funnel is not None:
                funnel.dropped_height += 1
                funnel.drop_reasons["candidate_height_gate"] += 1
            continue
        if distance > maximum_detection_range_m:
            if funnel is not None:
                funnel.drop_reasons["candidate_range_gate"] += 1
            continue
        result.append(BasePoint3D(int(stamp_ns), str(source_sensor), int(point_index), x_base, y_base, z_base, distance))
        if funnel is not None:
            funnel.accepted_points += 1
    return tuple(result)


def cluster_base_pointcloud(
    points: Sequence[BasePoint3D],
    *,
    voxel_size_m: float = 0.25,
    neighbor_radius_m: float = 0.85,
    minimum_support: int = 8,
    maximum_clusters: int = 80,
    organized_structure_only: bool = False,
    maximum_vertical_range_delta_m: float = 0.30,
    maximum_vertical_surface_normal_up_abs: float = 0.70,
    minimum_vertical_consistent_neighbors: int = 1,
    entity_evidence: bool = False,
    center_mode: str = "mean",
    center_radial_offset_m: float = 0.0,
    funnel: PointCloudFunnel | None = None,
) -> tuple[CandidateObservation, ...]:
    """在三维体素邻域上聚类，避免把多线点云误当成单行 ranges。"""

    if voxel_size_m <= 0.0 or neighbor_radius_m <= 0.0 or minimum_support <= 0:
        raise ValueError("POINTCLOUD_CLUSTER_PARAMETER_INVALID")
    if center_mode not in {"mean", "pca_bbox"} or not math.isfinite(center_radial_offset_m):
        raise ValueError("POINTCLOUD_CENTER_PARAMETER_INVALID")
    if organized_structure_only:
        points = tuple(
            point for point in points
            if point.vertical_consistent_neighbor_count >= minimum_vertical_consistent_neighbors
            and (
                (
                    math.isfinite(point.organized_normal_up_abs)
                    and point.organized_normal_up_abs <= maximum_vertical_surface_normal_up_abs
                )
                or (
                    math.isfinite(point.min_vertical_range_delta_m)
                    and point.min_vertical_range_delta_m <= maximum_vertical_range_delta_m
                )
            )
        )
    if not points:
        return ()
    # 生产路径使用 NumPy 做体素化，避免每个点和每个体素都经过 Python 嵌套循环。
    coordinates = np.asarray([(point.x_m, point.y_m, point.z_m) for point in points], dtype=np.float64)
    ranges = np.asarray([point.range_m for point in points], dtype=np.float64)
    keys = np.floor(coordinates / float(voxel_size_m)).astype(np.int64)
    unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    cell_count = int(unique_keys.shape[0])
    if funnel is not None:
        funnel.voxel_cells += cell_count
    order = np.argsort(inverse, kind="stable")
    boundaries = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    cell_point_indices = [order[int(boundaries[index]):int(boundaries[index + 1])] for index in range(cell_count)]
    cell_centers = np.zeros((cell_count, 3), dtype=np.float64)
    np.add.at(cell_centers, inverse, coordinates)
    cell_centers /= counts[:, None]
    # 按实际体素中心距离连边。旧网格偏移门会漏掉跨两格但实际相距很近的点。
    edges = cKDTree(cell_centers).query_pairs(float(neighbor_radius_m), output_type="ndarray")
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(cell_count, cell_count))
    _, labels = connected_components(graph, directed=False)
    grouped_cells = [np.flatnonzero(labels == label).tolist() for label in np.unique(labels)]
    grouped_cells.sort(key=lambda group: -sum(int(counts[cell]) for cell in group))
    if funnel is not None:
        funnel.clusters_seen += len(grouped_cells)
    fast_result: list[CandidateObservation] = []
    source = str(points[0].source_sensor)
    stamp = int(points[0].stamp_ns)
    for cluster_index, group in enumerate(grouped_cells[: int(maximum_clusters)]):
        point_indices = np.concatenate([cell_point_indices[cell] for cell in group])
        support = int(point_indices.size)
        if support < int(minimum_support):
            if funnel is not None:
                funnel.clusters_rejected_support += 1
                funnel.drop_reasons["cluster_too_few_points"] += 1
            continue
        cluster_xyz = coordinates[point_indices]
        mean_xyz = np.mean(cluster_xyz, axis=0)
        center_x, center_y, center_z = mean_xyz
        if center_mode == "pca_bbox" and support >= 3:
            centered_xy = cluster_xyz[:, :2] - mean_xyz[:2]
            planar_covariance = centered_xy.T @ centered_xy / float(support)
            _, planar_axes = np.linalg.eigh(planar_covariance)
            projected_xy = centered_xy @ planar_axes
            projected_midpoint = 0.5 * (
                np.min(projected_xy, axis=0) + np.max(projected_xy, axis=0)
            )
            center_x, center_y = mean_xyz[:2] + projected_midpoint @ planar_axes.T
        center_range = math.hypot(float(center_x), float(center_y))
        if center_range > 1e-9 and center_radial_offset_m != 0.0:
            center_x += center_radial_offset_m * float(center_x) / center_range
            center_y += center_radial_offset_m * float(center_y) / center_range
        radial = np.hypot(cluster_xyz[:, 0] - center_x, cluster_xyz[:, 1] - center_y)
        radius = max(0.20, float(np.max(radial)) if radial.size else 0.20)
        spread = math.sqrt(float(np.mean(radial * radial)))
        if support >= 2:
            centered_xy = cluster_xyz[:, :2] - mean_xyz[:2]
            planar_covariance = centered_xy.T @ centered_xy / float(support)
            planar_eigenvalues = np.maximum(0.0, np.linalg.eigvalsh(planar_covariance))
            horizontal_minor_extent = 4.0 * math.sqrt(float(planar_eigenvalues[0]))
            horizontal_major_extent = 4.0 * math.sqrt(float(planar_eigenvalues[1]))
        else:
            horizontal_minor_extent = 0.0
            horizontal_major_extent = 0.0
        z_min = float(np.min(cluster_xyz[:, 2]))
        z_max = float(np.max(cluster_xyz[:, 2]))
        vertical_extent = z_max - z_min
        vertical_layers = len({int(keys[int(index), 2]) for index in point_indices})
        if support >= 3:
            centered_xyz = cluster_xyz - np.mean(cluster_xyz, axis=0)
            covariance_xyz = centered_xyz.T @ centered_xyz / float(support)
            _, eigenvectors_xyz = np.linalg.eigh(covariance_xyz)
            surface_normal_up_abs = abs(float(eigenvectors_xyz[2, 0]))
        else:
            surface_normal_up_abs = 1.0
        organized_normals = np.asarray([points[int(index)].organized_normal_up_abs for index in point_indices])
        organized_normal_valid = np.isfinite(organized_normals)
        organized_normal_valid_fraction = float(np.mean(organized_normal_valid))
        vertical_surface_fraction = float(np.mean(organized_normals[organized_normal_valid] < 0.70)) if organized_normal_valid.any() else 0.0
        vertical_deltas = np.asarray([points[int(index)].min_vertical_range_delta_m for index in point_indices])
        finite_vertical_deltas = np.isfinite(vertical_deltas)
        vertical_stack_fraction = float(np.mean(vertical_deltas[finite_vertical_deltas] <= 0.75)) if finite_vertical_deltas.any() else 0.0
        mean_vertical_delta = float(np.mean(np.minimum(vertical_deltas[finite_vertical_deltas], 5.0))) if finite_vertical_deltas.any() else 5.0
        position_std = max(0.08, spread / math.sqrt(float(max(1, support))))
        range_mean = float(np.mean(ranges[point_indices]))
        confidence = min(0.98, max(0.05, (support / (support + 12.0)) * math.exp(-position_std / 2.0)))
        fast_result.append(
            CandidateObservation(
                stamp_ns=stamp,
                observation_id=f"pc:{source}:{stamp}:{cluster_index}",
                x_m=float(center_x),
                y_m=float(center_y),
                radius_m=float(radius),
                support=support,
                source_sensors=tuple(sorted({points[int(index)].source_sensor for index in point_indices})),
                position_std_m=float(position_std),
                range_m=range_mean,
                confidence=float(confidence),
                source_beam_span=(0, 0),
                observation_stamps_ns=(stamp,),
                z_m=float(center_z),
                z_min_m=z_min,
                z_max_m=z_max,
                vertical_extent_m=vertical_extent,
                horizontal_major_extent_m=horizontal_major_extent,
                horizontal_minor_extent_m=horizontal_minor_extent,
                surface_normal_up_abs=surface_normal_up_abs,
                vertical_voxel_layers=vertical_layers,
                vertical_surface_fraction=vertical_surface_fraction,
                organized_vertical_stack_fraction=vertical_stack_fraction,
                mean_min_vertical_range_delta_m=mean_vertical_delta,
                organized_normal_valid_fraction=organized_normal_valid_fraction,
                entity_evidence=bool(entity_evidence),
            )
        )
    if funnel is not None:
        funnel.candidates_emitted += len(fast_result)
    return tuple(fast_result)

    # 保留下面的朴素实现作为审计参考；生产路径在上面已返回。
    cells: dict[tuple[int, int, int], list[BasePoint3D]] = defaultdict(list)
    for point in points:
        key = (
            math.floor(float(point.x_m) / float(voxel_size_m)),
            math.floor(float(point.y_m) / float(voxel_size_m)),
            math.floor(float(point.z_m) / float(voxel_size_m)),
        )
        cells[key].append(point)
    if funnel is not None:
        funnel.voxel_cells += len(cells)
    cell_centers = {
        key: (
            sum(item.x_m for item in values) / len(values),
            sum(item.y_m for item in values) / len(values),
            sum(item.z_m for item in values) / len(values),
        )
        for key, values in cells.items()
    }
    step = max(1, int(math.ceil(float(neighbor_radius_m) / float(voxel_size_m))))
    radius_sq = float(neighbor_radius_m) * float(neighbor_radius_m)
    offsets = [
        (dx, dy, dz)
        for dx in range(-step, step + 1)
        for dy in range(-step, step + 1)
        for dz in range(-step, step + 1)
        if (dx, dy, dz) != (0, 0, 0)
        and (dx * voxel_size_m) ** 2 + (dy * voxel_size_m) ** 2 + (dz * voxel_size_m) ** 2 <= radius_sq + 1e-12
    ]
    visited: set[tuple[int, int, int]] = set()
    groups: list[list[BasePoint3D]] = []
    for start in sorted(cells):
        if start in visited:
            continue
        visited.add(start)
        queue: deque[tuple[int, int, int]] = deque([start])
        group_cells: list[tuple[int, int, int]] = []
        while queue:
            current = queue.popleft()
            group_cells.append(current)
            cx, cy, cz = cell_centers[current]
            for dx, dy, dz in offsets:
                candidate = (current[0] + dx, current[1] + dy, current[2] + dz)
                if candidate in visited or candidate not in cells:
                    continue
                nx, ny, nz = cell_centers[candidate]
                if (cx - nx) ** 2 + (cy - ny) ** 2 + (cz - nz) ** 2 <= radius_sq:
                    visited.add(candidate)
                    queue.append(candidate)
        groups.append([point for key in group_cells for point in cells[key]])
    if funnel is not None:
        funnel.clusters_seen += len(groups)

    groups.sort(key=lambda group: (-len(group), min(item.point_index for item in group)))
    result: list[CandidateObservation] = []
    for cluster_index, group in enumerate(groups[: int(maximum_clusters)]):
        support = len(group)
        if support < int(minimum_support):
            if funnel is not None:
                funnel.clusters_rejected_support += 1
                funnel.drop_reasons["cluster_too_few_points"] += 1
            continue
        center_x = sum(item.x_m for item in group) / float(support)
        center_y = sum(item.y_m for item in group) / float(support)
        center_z = sum(item.z_m for item in group) / float(support)
        radial = [math.hypot(item.x_m - center_x, item.y_m - center_y) for item in group]
        radius = max(0.20, max(radial, default=0.20))
        spread = math.sqrt(sum(value * value for value in radial) / float(max(1, support)))
        position_std = max(0.08, spread / math.sqrt(float(max(1, support))))
        sources = tuple(sorted({str(item.source_sensor) for item in group}))
        range_mean = sum(item.range_m for item in group) / float(support)
        confidence = min(0.98, max(0.05, (support / (support + 12.0)) * math.exp(-position_std / 2.0)))
        result.append(
            CandidateObservation(
                stamp_ns=int(group[0].stamp_ns),
                observation_id=f"pc:{','.join(sources)}:{group[0].stamp_ns}:{cluster_index}",
                x_m=center_x,
                y_m=center_y,
                radius_m=radius,
                support=support,
                source_sensors=sources,
                position_std_m=position_std,
                range_m=range_mean,
                confidence=confidence,
                source_beam_span=(0, 0),
                observation_stamps_ns=(int(group[0].stamp_ns),),
                z_m=center_z,
            )
        )
    if funnel is not None:
        funnel.candidates_emitted += len(result)
    return tuple(result)


__all__ = [
    "BasePoint3D",
    "PointCloudFunnel",
    "cluster_base_pointcloud",
    "pointcloud_to_base_points",
    "organized_point_features",
    "organized_vertical_neighbor_counts",
]
