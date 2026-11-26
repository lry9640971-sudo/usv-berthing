"""靠泊感知的无 ROS 数学与消息载荷内核。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


LIDAR_TOPICS = (
    "/wamv/sensors/lidars/back_lidar_sensor/points",
    "/wamv/sensors/lidars/forward_lidar_sensor/points",
    "/wamv/sensors/lidars/front_lidar_sensor/points",
    "/wamv/sensors/lidars/left_lidar_sensor/points",
    "/wamv/sensors/lidars/right_lidar_sensor/points",
)
BERTH_CENTERS_DOCK_M = ((1.5, 3.0), (1.5, 9.0), (1.5, 15.0))
# 实际泊位内净宽 4 m；船长约 5.37 m，沿入口 +X 停靠，不能横向转 90°。
BERTH_YAW_DOCK_RAD = 0.0
WGS84_A_M = 6378137.0
WGS84_E2 = 6.6943799901413165e-3


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def quaternion_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    values = np.asarray((x, y, z, w), dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(values).all() or norm <= 1e-12:
        raise ValueError("IMU_QUATERNION_INVALID")
    x, y, z, w = values / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def leveling_rotation(roll: float, pitch: float) -> np.ndarray:
    # base->ENU 的旋转为 Rz(yaw)Ry(pitch)Rx(roll)；水平化保留 yaw 并去掉姿态倾斜。
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rotate_x = np.asarray(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)), dtype=np.float64)
    rotate_y = np.asarray(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)), dtype=np.float64)
    return rotate_y @ rotate_x


def load_extrinsics(path: str | Path) -> dict[str, np.ndarray]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    values = {
        str(topic): np.asarray(item["T_base_sensor"], dtype=np.float64)
        for topic, item in document["extrinsics"].items()
    }
    missing = [topic for topic in LIDAR_TOPICS if topic not in values]
    if missing:
        raise ValueError(f"LIDAR_EXTRINSICS_MISSING:{','.join(missing)}")
    for topic, transform in values.items():
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"LIDAR_EXTRINSIC_INVALID:{topic}")
    return values


def points_to_base(points_sensor: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_sensor, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return points
    return points @ transform[:3, :3].T + transform[:3, 3]


def aggregate_xy(points_xyz: np.ndarray, resolution_m: float = 0.20) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)
    keys = np.floor(points[:, :2] / float(resolution_m)).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices), :2]


def _ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    latitude = math.radians(float(latitude_deg))
    longitude = math.radians(float(longitude_deg))
    sine_latitude = math.sin(latitude)
    radius = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sine_latitude * sine_latitude)
    return np.asarray(
        (
            (radius + altitude_m) * math.cos(latitude) * math.cos(longitude),
            (radius + altitude_m) * math.cos(latitude) * math.sin(longitude),
            (radius * (1.0 - WGS84_E2) + altitude_m) * sine_latitude,
        ),
        dtype=np.float64,
    )


def project_enu(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    datum_latitude_deg: float,
    datum_longitude_deg: float,
    datum_altitude_m: float,
) -> np.ndarray:
    latitude0 = math.radians(float(datum_latitude_deg))
    longitude0 = math.radians(float(datum_longitude_deg))
    delta = _ecef(latitude_deg, longitude_deg, altitude_m) - _ecef(
        datum_latitude_deg, datum_longitude_deg, datum_altitude_m
    )
    slat, clat = math.sin(latitude0), math.cos(latitude0)
    slon, clon = math.sin(longitude0), math.cos(longitude0)
    return np.asarray(
        (
            -slon * delta[0] + clon * delta[1],
            -slat * clon * delta[0] - slat * slon * delta[1] + clat * delta[2],
            clat * clon * delta[0] + clat * slon * delta[1] + slat * delta[2],
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class BasePose:
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    roll_rad: float
    pitch_rad: float


def base_pose_from_navigation(
    fix: Iterable[float],
    rpy: Iterable[float],
    datum: Iterable[float],
    gps_lever_base_xy_m: Iterable[float],
) -> BasePose:
    latitude, longitude, altitude = (float(value) for value in fix)
    roll, pitch, yaw = (float(value) for value in rpy)
    datum_latitude, datum_longitude, datum_altitude = (float(value) for value in datum)
    enu = project_enu(latitude, longitude, altitude, datum_latitude, datum_longitude, datum_altitude)
    lever_x, lever_y = (float(value) for value in gps_lever_base_xy_m)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    lever_world_x = cosine * lever_x - sine * lever_y
    lever_world_y = sine * lever_x + cosine * lever_y
    return BasePose(
        float(enu[0] - lever_world_x),
        float(enu[1] - lever_world_y),
        float(enu[2]),
        yaw,
        roll,
        pitch,
    )


def world_pose_to_base(
    world_x: float,
    world_y: float,
    world_yaw: float,
    base_pose: BasePose,
) -> dict[str, float]:
    dx, dy = float(world_x) - base_pose.x_m, float(world_y) - base_pose.y_m
    cosine, sine = math.cos(base_pose.yaw_rad), math.sin(base_pose.yaw_rad)
    return {
        "x": cosine * dx + sine * dy,
        "y": -sine * dx + cosine * dy,
        "yaw": wrap_angle(float(world_yaw) - base_pose.yaw_rad),
    }


def base_pose_to_world(base_x: float, base_y: float, base_yaw: float, vehicle: BasePose) -> tuple[float, float, float]:
    cosine, sine = math.cos(vehicle.yaw_rad), math.sin(vehicle.yaw_rad)
    return (
        vehicle.x_m + cosine * float(base_x) - sine * float(base_y),
        vehicle.y_m + sine * float(base_x) + cosine * float(base_y),
        wrap_angle(vehicle.yaw_rad + float(base_yaw)),
    )


def child_pose(parent: tuple[float, float, float], local: tuple[float, float, float]) -> tuple[float, float, float]:
    px, py, pyaw = parent
    lx, ly, lyaw = local
    cosine, sine = math.cos(pyaw), math.sin(pyaw)
    return (
        px + cosine * lx - sine * ly,
        py + sine * lx + cosine * ly,
        wrap_angle(pyaw + lyaw),
    )


def points_in_dock_frame(points_base: np.ndarray, dock_base_pose: tuple[float, float, float]) -> np.ndarray:
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    tx, ty, yaw = dock_base_pose
    cosine, sine = math.cos(yaw), math.sin(yaw)
    delta_x = points[:, 0] - tx
    delta_y = points[:, 1] - ty
    output = points.copy()
    output[:, 0] = cosine * delta_x + sine * delta_y
    output[:, 1] = -sine * delta_x + cosine * delta_y
    return output


def infer_bay_occupancy(
    points_base: np.ndarray,
    dock_base_pose: tuple[float, float, float],
    *,
    minimum_cells: int = 5,
    ray_origins_base: np.ndarray | None = None,
    ownship_mask: np.ndarray | None = None,
) -> list[dict]:
    """命中支持占用；只有有限回波射线覆盖内部才支持空闲，未观测保持未知。"""

    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return [{"status": "UNKNOWN", "confidence": None, "cells": 0} for _ in BERTH_CENTERS_DOCK_M]
    finite = np.isfinite(points).all(axis=1)
    self_mask = (np.abs(points[:, 0]) <= 3.2) & (np.abs(points[:, 1]) <= 1.8) & (points[:, 2] <= 2.8)
    if ownship_mask is not None:
        self_mask = np.asarray(ownship_mask, dtype=bool)
        if self_mask.shape != finite.shape:
            raise ValueError("BAY_SELF_MASK_SHAPE_INVALID")
    eligible = finite & ~self_mask
    dock_points = points_in_dock_frame(points[eligible], dock_base_pose)
    origins = None
    if ray_origins_base is not None:
        values = np.asarray(ray_origins_base, dtype=float).reshape(-1, 3)
        if values.shape != points.shape:
            raise ValueError("BAY_RAY_ORIGIN_SHAPE_INVALID")
        origins = points_in_dock_frame(values[eligible], dock_base_pose)
    results = []
    for _, center_y in BERTH_CENTERS_DOCK_M:
        interior = (
            (dock_points[:, 0] >= -0.25)
            & (dock_points[:, 0] <= 4.25)
            & (np.abs(dock_points[:, 1] - center_y) <= 1.45)
            & (dock_points[:, 2] >= 0.18)
            & (dock_points[:, 2] <= 2.8)
        )
        candidates = dock_points[interior]
        if len(candidates):
            keys = np.floor(candidates[:, :2] / 0.25).astype(np.int64)
            occupied_cells = int(len(np.unique(keys, axis=0)))
        else:
            occupied_cells = 0
        if occupied_cells >= int(minimum_cells):
            confidence = min(0.98, 0.60 + 0.04 * (occupied_cells - int(minimum_cells)))
            results.append({"status": "OCCUPIED", "confidence": float(confidence), "cells": occupied_cells})
        else:
            coverage = 0.
            if origins is not None and len(origins):
                # 在泊位内部多个位置核对真正穿过的射线，NaN/Inf 不能作为无遮挡证据。
                xs, ys = np.meshgrid(np.arange(.25, 4.01, .5), center_y + np.arange(-1.25, 1.26, .5))
                cells = np.column_stack((xs.ravel(), ys.ravel()))
                # 先剔除不可能经过该泊位的射线，仍保留真实线段覆盖条件。
                segment_min, segment_max = np.minimum(dock_points, origins), np.maximum(dock_points, origins)
                possible = ((segment_max[:, 0] >= .05) & (segment_min[:, 0] <= 4.25)
                            & (segment_max[:, 1] >= center_y - 1.45) & (segment_min[:, 1] <= center_y + 1.45)
                            & (segment_max[:, 2] >= .18) & (segment_min[:, 2] <= 2.8))
                selected_origins, selected_points = origins[possible], dock_points[possible]
                vectors = selected_points - selected_origins
                square = np.sum(vectors[:, :2] ** 2, axis=1)
                safe_square = np.maximum(square, 1e-9)
                free = []
                for cell in cells:
                    offset = cell - selected_origins[:, :2]
                    dot = np.sum(offset * vectors[:, :2], axis=1)
                    fraction = dot / safe_square
                    perpendicular_square = np.maximum(0., np.sum(offset ** 2, axis=1) - dot * fraction)
                    height = selected_origins[:, 2] + fraction * vectors[:, 2]
                    free.append(bool(np.any((fraction > 0.) & (fraction < .98) & (perpendicular_square <= .20 ** 2)
                                            & (height >= .18) & (height <= 2.8))))
                coverage = float(np.mean(free))
            if occupied_cells == 0 and coverage >= .65:
                results.append({"status": "FREE", "confidence": min(.95, .60 + .35 * coverage),
                                "cells": 0, "observed_fraction": coverage})
            else:
                results.append({"status": "UNKNOWN", "confidence": None, "cells": occupied_cells,
                                "observed_fraction": coverage})
    return results


def pose_confidence(geometry_score: float, support: float, threshold: float) -> float:
    score_term = max(0.0, min(1.0, 1.0 - float(geometry_score) / max(1e-6, float(threshold))))
    return float(max(0.0, min(0.99, 0.20 + 0.40 * score_term + 0.40 * float(support))))


def covariance_xyyaw(residual_m: float, confidence: float) -> list[float]:
    position_variance = max(0.04, min(4.0, float(residual_m) ** 2 + (1.0 - confidence) * 0.25))
    yaw_variance = max(math.radians(0.5) ** 2, (1.0 - confidence) * math.radians(8.0) ** 2)
    return [position_variance, 0.0, 0.0, 0.0, position_variance, 0.0, 0.0, 0.0, yaw_variance]


def make_berth_candidates(
    dock_pose: tuple[float, float, float],
    occupancies: list[dict],
    *,
    valid: bool,
    confidence: float,
    covariance: list[float],
    reason: str,
    entrance_offset_m: float = -8.0,
) -> list[dict]:
    output = []
    for index, (_, center_y) in enumerate(BERTH_CENTERS_DOCK_M, start=1):
        final_pose = child_pose(dock_pose, (1.5, center_y, BERTH_YAW_DOCK_RAD))
        # 入口与最终姿态沿实际可容纳船体的 dock +X 方向。
        entrance_pose = child_pose(dock_pose, (float(entrance_offset_m), center_y, 0.0))
        corridor_local = (
            (float(entrance_offset_m), center_y - 1.5, 0.0),
            (2.25, center_y - 1.5, 0.0),
            (2.25, center_y + 1.5, 0.0),
            (float(entrance_offset_m), center_y + 1.5, 0.0),
        )
        corridor = [list(child_pose(dock_pose, point)[:2]) for point in corridor_local]
        occupancy = occupancies[index - 1] if valid else {"status": "UNKNOWN", "confidence": None, "cells": 0}
        candidate_valid = bool(valid and occupancy["status"] in {"FREE", "OCCUPIED"})
        pose_dict = lambda pose: {"x": float(pose[0]), "y": float(pose[1]), "yaw": float(pose[2])}
        output.append(
            {
                "candidate_id": f"dock_bay_{index}",
                "bay_id": index,
                "global_id_resolved": True,
                "entry_pose": pose_dict(entrance_pose),
                "entrance_pose": pose_dict(entrance_pose),
                "final_pose": pose_dict(final_pose),
                "approach_corridor_xy": corridor,
                "occupancy_status": occupancy["status"],
                "occupancy_confidence": occupancy["confidence"],
                "occupancy_evidence_cells": int(occupancy["cells"]),
                "pose_confidence": float(confidence),
                "confidence": float(confidence),
                "covariance_xyyaw": list(covariance),
                "valid": candidate_valid,
                "reason": reason if candidate_valid else (reason or "DOCK_POSE_INVALID"),
            }
        )
    return output


__all__ = [
    "BERTH_CENTERS_DOCK_M",
    "BasePose",
    "LIDAR_TOPICS",
    "aggregate_xy",
    "base_pose_from_navigation",
    "base_pose_to_world",
    "covariance_xyyaw",
    "infer_bay_occupancy",
    "leveling_rotation",
    "load_extrinsics",
    "make_berth_candidates",
    "points_to_base",
    "pose_confidence",
    "quaternion_to_rpy",
    "world_pose_to_base",
    "wrap_angle",
    "yaw_to_quaternion",
]
