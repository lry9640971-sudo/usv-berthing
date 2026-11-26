"""动态船轨迹的几何语义特征和稳定对象记忆。

运行时只读取候选检测、定位状态和 tracker 输出。模型参数来自离线开发数据，
本模块不订阅、读取或推断 Gazebo truth。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2

from .s3_candidate_repair import CandidateObservation
from .s3_tracker_repair import TrackEstimate, exact_one_to_one_assignment


LEGACY_SEMANTIC_FEATURE_NAMES = (
    "hit_count", "confidence", "measurement_count", "entity_evidence_count",
    "entity_evidence_fraction", "strong_fraction", "multi_sensor_fraction",
    "shape_fraction", "max_support", "max_sources", "max_vertical_extent",
    "max_major_extent", "max_minor_extent", "max_radius", "mean_range",
    "minimum_range", "maximum_range", "mean_log_support", "mean_vertical_extent",
    "mean_minor_extent", "mean_normal_up", "mean_vertical_surface",
    "mean_vertical_stack", "mean_vertical_delta", "mean_organized_normal_valid",
    "max_vertical_surface", "max_vertical_stack", "max_organized_normal_valid",
    "mean_z", "std_z", "max_z", "recent_motion_span_s",
    "recent_motion_speed_mps", "recent_path_efficiency", "recent_line_fit_rms_m",
    "recent_range_span_s", "recent_range_change_m", "recent_range_std_m",
    "recent_range_rate_mps", "recent_range_path_m", "recent_range_efficiency",
    "ownship_speed", "velocity_ego_delta", "velocity_ego_alignment",
    "relative_radial_velocity", "relative_tangential_velocity",
    "relative_tangential_speed", "range_rate_residual", "track_distance_to_ego",
    "range_position_residual", "recent_observation_count",
    "recent_entity_evidence_count", "recent_entity_evidence_fraction",
    "recent_shape_fraction", "recent_multi_sensor_fraction", "recent_max_support",
    "recent_max_sources", "recent_max_vertical_extent", "recent_max_major_extent",
    "recent_max_minor_extent", "recent_max_radius", "recent_mean_vertical_extent",
    "recent_mean_minor_extent", "recent_mean_vertical_surface",
    "recent_mean_vertical_stack", "recent_mean_vertical_delta",
    "recent_mean_organized_normal_valid", "recent_mean_z", "recent_max_z",
    "recent_mean_range", "mean_acceleration", "max_acceleration",
    "recent_velocity_std_mps", "recent_speed_std_mps",
    "recent_acceleration_mean_mps2", "recent_acceleration_p95_mps2",
    "recent_direction_consistency", "speed",
)

SEMANTIC_FEATURE_NAMES = LEGACY_SEMANTIC_FEATURE_NAMES + (
    "causal_background_frames", "causal_background_fraction",
)

_LOG1P_FEATURES = {"hit_count", "measurement_count", "entity_evidence_count", "max_support"}


def semantic_feature_vector(
    row: dict[str, float | int],
    feature_names: tuple[str, ...] = SEMANTIC_FEATURE_NAMES,
) -> list[float]:
    """生成与离线模型完全一致的有限特征向量。"""
    values: list[float] = []
    for name in feature_names:
        value = float(row.get(name, 0.0))
        if not math.isfinite(value):
            value = 0.0
        if name in _LOG1P_FEATURES:
            value = math.log1p(max(0.0, value))
        elif name == "recent_line_fit_rms_m":
            value = min(value, 5.0)
        values.append(value)
    return values


_VELOCITY_CONFIDENCE_RADIUS = math.sqrt(float(chi2.ppf(.95, 2)))


def uncertainty_motion_features(observations: list[CandidateObservation]) -> dict[str, float]:
    """把可见表面质心偏移计入运动拟合，不把固定厘米残差当成船体证据。"""
    if len(observations) < 3:
        return {}
    times = np.asarray([item.stamp_ns for item in observations], dtype=np.float64)
    times = (times - times[0]) / 1e9
    if np.ptp(times) <= 0:
        return {}
    design = np.column_stack((np.ones(len(times)), times))
    xy = np.asarray([[item.x_m, item.y_m] for item in observations])
    # 包围半径内的均匀表面偏移，加上候选本身的位置测量方差。
    variance = np.asarray([max(.05, item.position_std_m) ** 2 + item.radius_m ** 2 / 3.
                           for item in observations])
    weight = 1. / variance
    covariance = np.linalg.inv(design.T @ (weight[:, None] * design))
    fitted = covariance @ (design.T @ (weight[:, None] * xy))
    residual = xy - design @ fitted
    statistic = float(np.sum(weight * np.sum(residual ** 2, axis=1)))
    degrees = 2 * (len(times) - 2)
    slope_std = math.sqrt(float(covariance[1, 1]) * max(1., statistic / degrees))
    speed = float(np.linalg.norm(fitted[1]))
    return {"motion_fitted_speed_mps": speed,
            "motion_speed_lower95_mps": max(0., speed - _VELOCITY_CONFIDENCE_RADIUS * slope_std),
            "motion_residual_probability": float(chi2.sf(statistic, degrees)),
            "motion_reduced_chi2": statistic / degrees}


def motion_supported_semantic_probability(
    row: dict[str, float | int],
    model_probability: float,
    *,
    classification: str = "UNKNOWN",
    minimum_observations: int = 6,
    minimum_span_s: float = 1.00,
    minimum_speed_mps: float = 0.55,
    minimum_path_efficiency: float = 0.90,
    maximum_line_fit_rms_m: float = 0.20,
    near_range_m: float = 20.0,
    near_minimum_observations: int = 4,
    near_minimum_span_s: float = 0.60,
    near_minimum_speed_mps: float = 0.45,
    near_minimum_path_efficiency: float = 0.93,
    near_maximum_line_fit_rms_m: float = 0.12,
    supported_probability: float = 0.75,
    minimum_motion_birth_range_m: float = 5.0,
) -> tuple[float, bool]:
    """用连续 ENU 运动证据补足稀疏点云缺少外形证据的情况。

    保留连续低残差通路。可见船表面跳动时，成熟船级轨迹还可由原语义模型、
    重复跨雷达印证与计入表面误差的显著运动共同确认，不强求固定厘米残差。
    单帧、静态抖动及不相容关联不因此放行；原因果背景门继续生效。
    """

    probability = min(1.0, max(0.0, float(model_probability)))
    general_supported = (
        int(row.get("recent_observation_count", 0)) >= int(minimum_observations)
        and float(row.get("recent_motion_span_s", 0.0)) >= float(minimum_span_s)
        and float(row.get("recent_motion_speed_mps", 0.0)) >= float(minimum_speed_mps)
        and float(row.get("recent_path_efficiency", 0.0)) >= float(minimum_path_efficiency)
        and float(row.get("recent_line_fit_rms_m", math.inf)) <= float(maximum_line_fit_rms_m)
    )
    near_supported = (
        float(row.get("track_distance_to_ego", math.inf)) <= float(near_range_m)
        and int(row.get("recent_observation_count", 0)) >= int(near_minimum_observations)
        and float(row.get("recent_motion_span_s", 0.0)) >= float(near_minimum_span_s)
        and float(row.get("recent_motion_speed_mps", 0.0)) >= float(near_minimum_speed_mps)
        and float(row.get("recent_path_efficiency", 0.0)) >= float(near_minimum_path_efficiency)
        and float(row.get("recent_line_fit_rms_m", math.inf)) <= float(near_maximum_line_fit_rms_m)
    )
    # 五路点云偶尔会把本船边缘或紧邻船体的水花串成一条随本船移动的轨迹。
    # 这种回波在绝对 ENU 中看似运动稳定，却不能仅凭运动证据出生。真实船在
    # 五米外即可建立对象，之后仍可正常跟踪到近距离；高概率外形模型不受此门限制。
    far_enough_for_motion_birth = (
        float(row.get("track_distance_to_ego", math.inf))
        > float(minimum_motion_birth_range_m)
    )
    # 部分船体回波会在同一船的不同表面间跳动。必须有重复实体证据，且
    # 计入该观测误差后速度下界仍足够大、残差与模型相容，才补强出生概率。
    entity_supported = (
        int(row.get("recent_observation_count", 0)) >= int(minimum_observations)
        and int(row.get("recent_entity_evidence_count", 0)) >= 2
        and float(row.get("recent_motion_span_s", 0.0)) >= float(minimum_span_s)
        and float(row.get("motion_speed_lower95_mps", 0.0)) >= float(minimum_speed_mps)
        and float(row.get("motion_residual_probability", 0.0)) >= .01
    )
    # 两场实际输入对照中，较低语义门会放行静态表面关联。
    # 稀疏通路保留较强语义证据、两次跨雷达实测和原运动一致性标准。
    corroborated_sparse_supported = (
        str(classification) == "BOAT"
        and probability >= .80
        and int(row.get("recent_observation_count", 0)) >= int(minimum_observations)
        and float(row.get("recent_motion_span_s", 0.0)) >= float(minimum_span_s)
        and float(row.get("recent_multi_sensor_fraction", 0.0))
        * int(row.get("recent_observation_count", 0)) >= 2. - 1e-9
        and float(row.get("motion_speed_lower95_mps", 0.0)) >= float(minimum_speed_mps)
        and float(row.get("motion_residual_probability", 0.0)) >= .01
    )
    supported = (general_supported or near_supported or entity_supported
                 or corroborated_sparse_supported) and far_enough_for_motion_birth
    if supported:
        return max(probability, float(supported_probability)), True
    return probability, False


@dataclass
class TrackEvidenceHistory:
    """聚合同一底层轨迹的实体形状、相对运动和速度稳定性。"""

    measurement_count: int = 0
    entity_evidence_count: int = 0
    strong_count: int = 0
    multi_sensor_count: int = 0
    shape_count: int = 0
    max_support: int = 0
    max_sources: int = 0
    max_vertical_extent: float = 0.0
    max_major_extent: float = 0.0
    max_minor_extent: float = 0.0
    max_radius: float = 0.0
    range_sum: float = 0.0
    minimum_range: float = 999.0
    maximum_range: float = 0.0
    support_log_sum: float = 0.0
    vertical_sum: float = 0.0
    minor_sum: float = 0.0
    normal_up_sum: float = 0.0
    vertical_surface_sum: float = 0.0
    vertical_stack_sum: float = 0.0
    vertical_delta_sum: float = 0.0
    organized_normal_valid_sum: float = 0.0
    max_vertical_surface: float = 0.0
    max_vertical_stack: float = 0.0
    max_organized_normal_valid: float = 0.0
    z_sum: float = 0.0
    z_squared_sum: float = 0.0
    max_z: float = -999.0
    previous_velocity: tuple[float, float] | None = None
    acceleration_sum: float = 0.0
    acceleration_max: float = 0.0
    velocity_updates: int = 0
    positions: deque[tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=80))
    ranges: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=80))
    observations: deque[CandidateObservation] = field(default_factory=lambda: deque(maxlen=80))
    velocities: deque[tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=80))

    def add_observation(self, observation: CandidateObservation) -> None:
        sources = len(observation.source_sensors)
        self.measurement_count += 1
        self.entity_evidence_count += int(observation.entity_evidence)
        self.strong_count += int(observation.support >= 3 and sources >= 2)
        self.multi_sensor_count += int(sources >= 2)
        self.shape_count += int(
            observation.vertical_extent_m >= 0.10
            and observation.horizontal_minor_extent_m >= 0.10
        )
        self.max_support = max(self.max_support, observation.support)
        self.max_sources = max(self.max_sources, sources)
        self.max_vertical_extent = max(self.max_vertical_extent, observation.vertical_extent_m)
        self.max_major_extent = max(self.max_major_extent, observation.horizontal_major_extent_m)
        self.max_minor_extent = max(self.max_minor_extent, observation.horizontal_minor_extent_m)
        self.max_radius = max(self.max_radius, observation.radius_m)
        self.range_sum += observation.range_m
        self.minimum_range = min(self.minimum_range, observation.range_m)
        self.maximum_range = max(self.maximum_range, observation.range_m)
        self.support_log_sum += math.log1p(observation.support)
        self.vertical_sum += observation.vertical_extent_m
        self.minor_sum += observation.horizontal_minor_extent_m
        self.normal_up_sum += observation.surface_normal_up_abs
        self.vertical_surface_sum += observation.vertical_surface_fraction
        self.vertical_stack_sum += observation.organized_vertical_stack_fraction
        self.vertical_delta_sum += observation.mean_min_vertical_range_delta_m
        self.organized_normal_valid_sum += observation.organized_normal_valid_fraction
        self.max_vertical_surface = max(self.max_vertical_surface, observation.vertical_surface_fraction)
        self.max_vertical_stack = max(self.max_vertical_stack, observation.organized_vertical_stack_fraction)
        self.max_organized_normal_valid = max(
            self.max_organized_normal_valid, observation.organized_normal_valid_fraction
        )
        self.z_sum += observation.z_m
        self.z_squared_sum += observation.z_m * observation.z_m
        self.max_z = max(self.max_z, observation.z_m)
        self.positions.append((int(observation.stamp_ns), float(observation.x_m), float(observation.y_m)))
        self.ranges.append((int(observation.stamp_ns), float(observation.range_m)))
        self.observations.append(observation)

    def add_velocity(self, stamp_ns: int, vx: float, vy: float, dt: float) -> None:
        current = (float(vx), float(vy))
        if self.previous_velocity is not None and dt > 1e-6:
            acceleration = math.hypot(
                current[0] - self.previous_velocity[0], current[1] - self.previous_velocity[1]
            ) / dt
            self.acceleration_sum += acceleration
            self.acceleration_max = max(self.acceleration_max, acceleration)
            self.velocity_updates += 1
        self.previous_velocity = current
        self.velocities.append((int(stamp_ns), current[0], current[1]))

    @staticmethod
    def _recent(items: Iterable[Any], last_stamp_ns: int, window_s: float = 2.0) -> list[Any]:
        cutoff = int(last_stamp_ns) - int(window_s * 1e9)
        return [item for item in items if int(item[0] if isinstance(item, tuple) else item.stamp_ns) >= cutoff]

    def _recent_motion(self, stamp_ns: int) -> dict[str, float]:
        rows = self._recent(self.positions, stamp_ns)
        empty = {
            "recent_motion_span_s": 0.0, "recent_motion_speed_mps": 0.0,
            "recent_path_efficiency": 0.0, "recent_line_fit_rms_m": 999.0,
        }
        if len(rows) < 3:
            return empty
        stamps = np.asarray([row[0] for row in rows], dtype=np.float64)
        xy = np.asarray([[row[1], row[2]] for row in rows], dtype=np.float64)
        times = (stamps - stamps[0]) / 1e9
        span = float(times[-1])
        if span <= 0.0:
            return empty
        design = np.column_stack((np.ones(len(times)), times))
        fitted = design @ np.linalg.lstsq(design, xy, rcond=None)[0]
        residual = float(np.sqrt(np.mean(np.sum((xy - fitted) ** 2, axis=1))))
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        path = float(np.sum(steps))
        displacement = float(np.linalg.norm(xy[-1] - xy[0]))
        result = {
            "recent_motion_span_s": span,
            "recent_motion_speed_mps": displacement / span,
            "recent_path_efficiency": displacement / max(1e-6, path),
            "recent_line_fit_rms_m": residual,
        }
        observations = self._recent(self.observations, stamp_ns)
        if len(observations) >= 6 and span >= 1.:
            result.update(uncertainty_motion_features(observations))
        return result

    def _recent_range_motion(self, stamp_ns: int) -> dict[str, float]:
        rows = self._recent(self.ranges, stamp_ns)
        empty = {
            "recent_range_span_s": 0.0, "recent_range_change_m": 0.0,
            "recent_range_std_m": 0.0, "recent_range_rate_mps": 0.0,
            "recent_range_path_m": 0.0, "recent_range_efficiency": 0.0,
        }
        if len(rows) < 3:
            return empty
        stamps = np.asarray([row[0] for row in rows], dtype=np.float64)
        values = np.asarray([row[1] for row in rows], dtype=np.float64)
        times = (stamps - stamps[0]) / 1e9
        span = float(times[-1])
        if span <= 0.0:
            return empty
        rate = float(np.linalg.lstsq(np.column_stack((np.ones(len(times)), times)), values, rcond=None)[0][1])
        change = float(abs(values[-1] - values[0]))
        path = float(np.sum(np.abs(np.diff(values))))
        return {
            "recent_range_span_s": span, "recent_range_change_m": change,
            "recent_range_std_m": float(np.std(values)), "recent_range_rate_mps": rate,
            "recent_range_path_m": path, "recent_range_efficiency": change / max(1e-6, path),
        }

    def _recent_shape(self, stamp_ns: int) -> dict[str, float | int]:
        rows = self._recent(self.observations, stamp_ns)
        if not rows:
            return {}
        n = len(rows)
        evidence_count = sum(int(row.entity_evidence) for row in rows)
        return {
            "recent_observation_count": n,
            "recent_entity_evidence_count": evidence_count,
            "recent_entity_evidence_fraction": evidence_count / n,
            "recent_shape_fraction": sum(int(row.vertical_extent_m >= .10 and row.horizontal_minor_extent_m >= .10) for row in rows) / n,
            "recent_multi_sensor_fraction": sum(int(len(row.source_sensors) >= 2) for row in rows) / n,
            "recent_max_support": max(row.support for row in rows),
            "recent_max_sources": max(len(row.source_sensors) for row in rows),
            "recent_max_vertical_extent": max(row.vertical_extent_m for row in rows),
            "recent_max_major_extent": max(row.horizontal_major_extent_m for row in rows),
            "recent_max_minor_extent": max(row.horizontal_minor_extent_m for row in rows),
            "recent_max_radius": max(row.radius_m for row in rows),
            "recent_mean_vertical_extent": sum(row.vertical_extent_m for row in rows) / n,
            "recent_mean_minor_extent": sum(row.horizontal_minor_extent_m for row in rows) / n,
            "recent_mean_vertical_surface": sum(row.vertical_surface_fraction for row in rows) / n,
            "recent_mean_vertical_stack": sum(row.organized_vertical_stack_fraction for row in rows) / n,
            "recent_mean_vertical_delta": sum(row.mean_min_vertical_range_delta_m for row in rows) / n,
            "recent_mean_organized_normal_valid": sum(row.organized_normal_valid_fraction for row in rows) / n,
            "recent_mean_z": sum(row.z_m for row in rows) / n,
            "recent_max_z": max(row.z_m for row in rows),
            "recent_mean_range": sum(row.range_m for row in rows) / n,
        }

    def _recent_velocity(self, stamp_ns: int) -> dict[str, float]:
        rows = self._recent(self.velocities, stamp_ns)
        empty = {
            "recent_velocity_std_mps": 0.0, "recent_speed_std_mps": 0.0,
            "recent_acceleration_mean_mps2": 0.0,
            "recent_acceleration_p95_mps2": 0.0, "recent_direction_consistency": 0.0,
        }
        if len(rows) < 3:
            return empty
        stamps = np.asarray([row[0] for row in rows], dtype=np.float64)
        vectors = np.asarray([[row[1], row[2]] for row in rows], dtype=np.float64)
        speeds = np.linalg.norm(vectors, axis=1)
        centered = vectors - np.mean(vectors, axis=0)
        dt = np.diff(stamps) / 1e9
        valid = dt > 1e-6
        accelerations = np.linalg.norm(np.diff(vectors, axis=0)[valid], axis=1) / dt[valid]
        moving = speeds > 1e-6
        if np.any(moving):
            unit = vectors[moving] / speeds[moving, None]
            direction_consistency = float(np.linalg.norm(np.mean(unit, axis=0)))
        else:
            direction_consistency = 0.0
        return {
            "recent_velocity_std_mps": float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))),
            "recent_speed_std_mps": float(np.std(speeds)),
            "recent_acceleration_mean_mps2": float(np.mean(accelerations)) if accelerations.size else 0.0,
            "recent_acceleration_p95_mps2": float(np.quantile(accelerations, .95)) if accelerations.size else 0.0,
            "recent_direction_consistency": direction_consistency,
        }

    def feature_row(
        self,
        stamp_ns: int,
        estimate: TrackEstimate,
        ownship_xyv: tuple[float, float, float, float],
    ) -> dict[str, float | int]:
        n = max(1, self.measurement_count)
        own_x, own_y, own_vx, own_vy = (float(value) for value in ownship_xyv)
        dx, dy = estimate.x - own_x, estimate.y - own_y
        distance = max(1e-6, math.hypot(dx, dy))
        relative_vx, relative_vy = estimate.vx - own_vx, estimate.vy - own_vy
        radial_velocity = (relative_vx * dx + relative_vy * dy) / distance
        tangential_velocity = (relative_vx * -dy + relative_vy * dx) / distance
        ownship_speed = math.hypot(own_vx, own_vy)
        alignment = (
            (estimate.vx * own_vx + estimate.vy * own_vy)
            / max(1e-6, estimate.speed_mps * ownship_speed)
        )
        result: dict[str, float | int] = {
            "hit_count": estimate.hit_count, "confidence": estimate.confidence,
            "measurement_count": self.measurement_count,
            "entity_evidence_count": self.entity_evidence_count,
            "entity_evidence_fraction": self.entity_evidence_count / n,
            "strong_fraction": self.strong_count / n,
            "multi_sensor_fraction": self.multi_sensor_count / n,
            "shape_fraction": self.shape_count / n,
            "max_support": self.max_support, "max_sources": self.max_sources,
            "max_vertical_extent": self.max_vertical_extent,
            "max_major_extent": self.max_major_extent, "max_minor_extent": self.max_minor_extent,
            "max_radius": self.max_radius, "mean_range": self.range_sum / n,
            "minimum_range": self.minimum_range, "maximum_range": self.maximum_range,
            "mean_log_support": self.support_log_sum / n,
            "mean_vertical_extent": self.vertical_sum / n, "mean_minor_extent": self.minor_sum / n,
            "mean_normal_up": self.normal_up_sum / n,
            "mean_vertical_surface": self.vertical_surface_sum / n,
            "mean_vertical_stack": self.vertical_stack_sum / n,
            "mean_vertical_delta": self.vertical_delta_sum / n,
            "mean_organized_normal_valid": self.organized_normal_valid_sum / n,
            "max_vertical_surface": self.max_vertical_surface,
            "max_vertical_stack": self.max_vertical_stack,
            "max_organized_normal_valid": self.max_organized_normal_valid,
            "mean_z": self.z_sum / n,
            "std_z": math.sqrt(max(0.0, self.z_squared_sum / n - (self.z_sum / n) ** 2)),
            "max_z": self.max_z,
            "mean_acceleration": self.acceleration_sum / max(1, self.velocity_updates),
            "max_acceleration": self.acceleration_max,
            "ownship_speed": ownship_speed,
            "velocity_ego_delta": math.hypot(relative_vx, relative_vy),
            "velocity_ego_alignment": alignment,
            "relative_radial_velocity": radial_velocity,
            "relative_tangential_velocity": tangential_velocity,
            "relative_tangential_speed": abs(tangential_velocity),
            "track_distance_to_ego": distance,
            "range_position_residual": abs(self.range_sum / n - distance),
            "speed": estimate.speed_mps,
        }
        result.update(self._recent_motion(stamp_ns))
        result.update(self._recent_range_motion(stamp_ns))
        result.update(self._recent_shape(stamp_ns))
        result.update(self._recent_velocity(stamp_ns))
        result["range_rate_residual"] = abs(
            float(result.get("recent_range_rate_mps", 0.0)) - radial_velocity
        )
        return result


class SemanticForestModel:
    """读取 sklearn 导出的树数组，以纯 Python 执行二分类概率推理。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != "phase16z_track_semantic_forest_v1":
            raise ValueError("SEMANTIC_MODEL_SCHEMA_INVALID")
        feature_names = tuple(payload.get("feature_names", ()))
        if feature_names != SEMANTIC_FEATURE_NAMES[:len(feature_names)]:
            raise ValueError("SEMANTIC_MODEL_FEATURE_CONTRACT_INVALID")
        self.feature_names = feature_names
        trees = payload.get("trees")
        if not isinstance(trees, list) or not trees:
            raise ValueError("SEMANTIC_MODEL_TREES_MISSING")
        self.trees = trees
        self.metadata = payload.get("metadata", {})

    @staticmethod
    def _tree_probability(tree: dict[str, Any], vector: list[float]) -> float:
        children_left = tree["children_left"]
        children_right = tree["children_right"]
        features = tree["feature"]
        thresholds = tree["threshold"]
        probabilities = tree["positive_probability"]
        node = 0
        while int(children_left[node]) >= 0:
            # sklearn 的树推理会先把输入压成 float32；边界附近也必须保持一致，
            # 否则导出的纯 Python 模型可能落入相邻叶节点。
            feature_value = float(np.float32(vector[int(features[node])]))
            node = int(children_left[node]) if feature_value <= float(thresholds[node]) else int(children_right[node])
        return float(probabilities[node])

    def predict_probability(self, row: dict[str, float | int]) -> float:
        vector = semantic_feature_vector(row, self.feature_names)
        value = sum(self._tree_probability(tree, vector) for tree in self.trees) / len(self.trees)
        return min(1.0, max(0.0, float(value)))


class CausalStaticBackgroundMap:
    """用过去候选的重复占据次数区分静态背景和移动目标。"""

    def __init__(
        self, *, cell_size_m: float = .5, query_radius_m: float = .8,
        history_window_s: float = 12.0, guard_s: float = 1.5,
    ) -> None:
        self.cell_size_m = float(cell_size_m)
        self.query_radius_m = float(query_radius_m)
        self.history_window_s = float(history_window_s)
        self.guard_s = float(guard_s)
        if min(self.cell_size_m, self.query_radius_m, self.history_window_s, self.guard_s) <= 0.0:
            raise ValueError("STATIC_BACKGROUND_PARAMETER_INVALID")
        if self.guard_s >= self.history_window_s:
            raise ValueError("STATIC_BACKGROUND_GUARD_INVALID")
        self.cells: dict[tuple[int, int], deque[int]] = defaultdict(deque)
        self.update_count = 0
        self.query_count = 0
        self.peak_cells = 0

    def _cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        return math.floor(float(x_m) / self.cell_size_m), math.floor(float(y_m) / self.cell_size_m)

    def features(self, stamp_ns: int, x_m: float, y_m: float) -> dict[str, float | int]:
        stamp = int(stamp_ns)
        oldest = stamp - int(self.history_window_s * 1e9)
        latest = stamp - int(self.guard_s * 1e9)
        center_x, center_y = self._cell(x_m, y_m)
        cell_radius = int(math.ceil(self.query_radius_m / self.cell_size_m))
        prior_stamps: set[int] = set()
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                if (dx * self.cell_size_m) ** 2 + (dy * self.cell_size_m) ** 2 > (self.query_radius_m + self.cell_size_m) ** 2:
                    continue
                history = self.cells.get((center_x + dx, center_y + dy))
                if not history:
                    continue
                while history and history[0] < oldest:
                    history.popleft()
                prior_stamps.update(value for value in history if value <= latest)
        self.query_count += 1
        frame_count = len(prior_stamps)
        return {
            "causal_background_frames": frame_count,
            "causal_background_fraction": min(
                1.0, frame_count / max(1.0, (self.history_window_s - self.guard_s) * 10.0)
            ),
        }

    def observe(self, stamp_ns: int, observations: Iterable[CandidateObservation]) -> None:
        stamp = int(stamp_ns)
        for cell in {self._cell(item.x_m, item.y_m) for item in observations}:
            history = self.cells[cell]
            if not history or history[-1] != stamp:
                history.append(stamp)
        self.update_count += 1
        self.peak_cells = max(self.peak_cells, len(self.cells))

    def status(self) -> dict[str, int | float]:
        return {
            "cell_size_m": self.cell_size_m,
            "query_radius_m": self.query_radius_m,
            "history_window_s": self.history_window_s,
            "guard_s": self.guard_s,
            "active_cells": len(self.cells),
            "peak_cells": self.peak_cells,
            "updates": self.update_count,
            "queries": self.query_count,
        }


class CausalVelocityRefiner:
    """用对象历史的圆弧方向和速度中位数减少转弯估计滞后。"""

    def __init__(
        self, *, history_window_s: float = 10.0, minimum_span_s: float = 3.0,
        circle_direction_weight: float = .5, history_speed_weight: float = .75,
        phase_lead_s: float = .25, fallback_heading_window_s: float = 3.0,
        fallback_minimum_span_s: float = .6, fallback_phase_lead_s: float = .5,
        fallback_heading_residual_limit_rad: float = .18,
        fallback_minimum_samples: int = 6,
    ) -> None:
        self.history_window_s = float(history_window_s)
        self.minimum_span_s = float(minimum_span_s)
        self.circle_direction_weight = float(circle_direction_weight)
        self.history_speed_weight = float(history_speed_weight)
        self.phase_lead_s = float(phase_lead_s)
        self.fallback_heading_window_s = float(fallback_heading_window_s)
        self.fallback_minimum_span_s = float(fallback_minimum_span_s)
        self.fallback_phase_lead_s = float(fallback_phase_lead_s)
        self.fallback_heading_residual_limit_rad = float(fallback_heading_residual_limit_rad)
        self.fallback_minimum_samples = int(fallback_minimum_samples)
        self.positions: dict[int, deque[tuple[int, float, float]]] = defaultdict(lambda: deque(maxlen=200))
        self.speeds: dict[int, deque[tuple[int, float]]] = defaultdict(lambda: deque(maxlen=200))
        self.velocities: dict[int, deque[tuple[int, float, float]]] = defaultdict(lambda: deque(maxlen=200))
        self.last_refined_velocity: dict[int, tuple[float, float]] = {}
        self.last_measured_state: dict[int, tuple[int, float, float, float, float]] = {}
        self.last_turn_rate: dict[int, float] = {}
        self.refined_count = 0
        self.arc_coast_count = 0

    def _heading_motion(self, history: deque[tuple[int, float, float]]):
        """圆拟合尚不可用时，从短期速度方向估计转率。"""

        if len(history) < self.fallback_minimum_samples:
            return None
        last_stamp = int(history[-1][0])
        cutoff = last_stamp - int(self.fallback_heading_window_s * 1e9)
        rows = [row for row in history if int(row[0]) >= cutoff]
        if len(rows) < self.fallback_minimum_samples:
            return None
        stamps = np.asarray([row[0] for row in rows], dtype=np.float64)
        span_s = float((stamps[-1] - stamps[0]) / 1e9)
        if span_s < self.fallback_minimum_span_s:
            return None
        velocities = np.asarray([[row[1], row[2]] for row in rows], dtype=np.float64)
        speeds = np.linalg.norm(velocities, axis=1)
        valid = speeds >= .4
        if int(np.count_nonzero(valid)) < self.fallback_minimum_samples:
            return None
        times = (stamps[valid] - stamps[valid][0]) / 1e9
        headings = np.unwrap(np.arctan2(velocities[valid, 1], velocities[valid, 0]))
        design = np.column_stack((np.ones(len(times)), times))
        try:
            intercept, omega = np.linalg.lstsq(design, headings, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        residual = headings - (intercept + omega * times)
        if float(np.quantile(np.abs(residual), .8)) > self.fallback_heading_residual_limit_rad:
            return None
        omega = float(np.clip(omega, -.3, .3))
        heading = float(headings[-1] + omega * self.fallback_phase_lead_s)
        speed = float(np.median(speeds[valid]))
        return np.asarray((math.cos(heading), math.sin(heading))), speed, omega

    def _circle_motion(self, history: deque[tuple[int, float, float]]):
        if len(history) < 12:
            return None
        stamps = np.asarray([row[0] for row in history], dtype=np.float64)
        if (stamps[-1] - stamps[0]) / 1e9 < self.minimum_span_s:
            return None
        xy = np.asarray([[row[1], row[2]] for row in history], dtype=np.float64)
        origin = np.mean(xy, axis=0)
        local = xy - origin
        design = np.column_stack((2.0 * local[:, 0], 2.0 * local[:, 1], np.ones(len(local))))
        target = np.sum(local * local, axis=1)
        try:
            solution = np.linalg.lstsq(design, target, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        center = origin + solution[:2]
        radii = np.linalg.norm(xy - center, axis=1)
        radius = float(np.median(radii))
        if not 4.0 <= radius <= 120.0 or float(np.quantile(np.abs(radii - radius), .8)) > .6:
            return None
        angles = np.unwrap(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
        times = (stamps - stamps[0]) / 1e9
        omega = float(np.linalg.lstsq(
            np.column_stack((np.ones(len(times)), times)), angles, rcond=None
        )[0][1])
        speed = abs(radius * omega)
        if not .5 <= speed <= 3.0 or not .015 <= abs(omega) <= .3:
            return None
        radial = xy[-1] - center
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 1e-6:
            return None
        tangent = np.asarray((-radial[1], radial[0])) / radial_norm
        tangent *= math.copysign(1.0, omega)
        return tangent, speed, omega

    def update(self, stamp_ns: int, estimates: Iterable[TrackEstimate]) -> tuple[TrackEstimate, ...]:
        stamp = int(stamp_ns)
        cutoff = stamp - int(self.history_window_s * 1e9)
        output: list[TrackEstimate] = []
        active_ids: set[int] = set()
        for estimate in estimates:
            track_id = int(estimate.track_id)
            active_ids.add(track_id)
            position_history = self.positions[track_id]
            speed_history = self.speeds[track_id]
            velocity_history = self.velocities[track_id]
            if int(estimate.missed_count) == 0:
                position_history.append((stamp, float(estimate.x), float(estimate.y)))
                if .3 <= float(estimate.speed_mps) <= 3.5:
                    speed_history.append((stamp, float(estimate.speed_mps)))
                    velocity_history.append((stamp, float(estimate.vx), float(estimate.vy)))
            while position_history and position_history[0][0] < cutoff:
                position_history.popleft()
            while speed_history and speed_history[0][0] < cutoff:
                speed_history.popleft()
            while velocity_history and velocity_history[0][0] < cutoff:
                velocity_history.popleft()
            if int(estimate.missed_count) != 0 and track_id in self.last_measured_state:
                measured_stamp, start_x, start_y, start_vx, start_vy = self.last_measured_state[track_id]
                elapsed = max(0.0, (stamp - measured_stamp) / 1e9)
                omega = self.last_turn_rate.get(track_id, 0.0)
                angle = omega * elapsed
                cosine, sine = math.cos(angle), math.sin(angle)
                vx = cosine * start_vx - sine * start_vy
                vy = sine * start_vx + cosine * start_vy
                if abs(omega) > 1e-6:
                    x = start_x + sine / omega * start_vx - (1.0 - cosine) / omega * start_vy
                    y = start_y + (1.0 - cosine) / omega * start_vx + sine / omega * start_vy
                    self.arc_coast_count += 1
                else:
                    x = start_x + elapsed * start_vx
                    y = start_y + elapsed * start_vy
                output.append(replace(estimate, x=x, y=y, vx=vx, vy=vy))
                continue
            fitted = self._circle_motion(position_history)
            raw = np.asarray((float(estimate.vx), float(estimate.vy)))
            raw_speed = float(np.linalg.norm(raw))
            if fitted is None:
                fitted = self._heading_motion(velocity_history)
            if fitted is None or raw_speed < .3:
                if int(estimate.missed_count) == 0:
                    self.last_measured_state[track_id] = (
                        stamp, float(estimate.x), float(estimate.y),
                        float(estimate.vx), float(estimate.vy),
                    )
                output.append(estimate)
                continue
            circle_direction, circle_speed, omega = fitted
            raw_direction = raw / raw_speed
            angle = omega * self.phase_lead_s
            cosine, sine = math.cos(angle), math.sin(angle)
            advanced = np.asarray((
                cosine * raw_direction[0] - sine * raw_direction[1],
                sine * raw_direction[0] + cosine * raw_direction[1],
            ))
            direction = (
                (1.0 - self.circle_direction_weight) * advanced
                + self.circle_direction_weight * circle_direction
            )
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-6:
                output.append(estimate)
                continue
            direction /= direction_norm
            historical_speed = float(np.median([value for _, value in speed_history]))
            base_speed = .25 * circle_speed + .75 * raw_speed
            speed = (
                (1.0 - self.history_speed_weight) * base_speed
                + self.history_speed_weight * historical_speed
            )
            vx, vy = float(speed * direction[0]), float(speed * direction[1])
            self.last_refined_velocity[track_id] = (vx, vy)
            self.last_turn_rate[track_id] = float(omega)
            self.last_measured_state[track_id] = (
                stamp, float(estimate.x), float(estimate.y), vx, vy,
            )
            self.refined_count += 1
            output.append(replace(estimate, vx=vx, vy=vy))
        for track_id in tuple(self.positions):
            if track_id not in active_ids:
                del self.positions[track_id]
                self.speeds.pop(track_id, None)
                self.velocities.pop(track_id, None)
                self.last_refined_velocity.pop(track_id, None)
                self.last_measured_state.pop(track_id, None)
                self.last_turn_rate.pop(track_id, None)
        return tuple(output)


class CausalFleetSpeedCalibrator:
    """用近期可靠对象的速度中位数抑制单条轨迹的表面测量偏差。"""

    def __init__(
        self, *, history_window_s: float = 10.0, prior_weight: float = .25,
        minimum_samples: int = 20, minimum_track_hits: int = 12,
        minimum_speed_mps: float = .8, maximum_speed_mps: float = 2.5,
    ) -> None:
        self.history_window_s = float(history_window_s)
        self.prior_weight = float(prior_weight)
        self.minimum_samples = int(minimum_samples)
        self.minimum_track_hits = int(minimum_track_hits)
        self.minimum_speed_mps = float(minimum_speed_mps)
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.samples: deque[tuple[int, float]] = deque(maxlen=4000)
        self.calibrated_count = 0
        self.last_prior_mps: float | None = None
        if (
            self.history_window_s <= 0.0
            or not 0.0 <= self.prior_weight <= 1.0
            or self.minimum_samples < 1
            or self.minimum_track_hits < 1
            or self.minimum_speed_mps <= 0.0
            or self.maximum_speed_mps <= self.minimum_speed_mps
        ):
            raise ValueError("FLEET_SPEED_CALIBRATOR_PARAMETER_INVALID")

    def update(self, stamp_ns: int, estimates: Iterable[TrackEstimate]) -> tuple[TrackEstimate, ...]:
        stamp = int(stamp_ns)
        rows = tuple(estimates)
        cutoff = stamp - int(self.history_window_s * 1e9)
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        for estimate in rows:
            speed = float(estimate.speed_mps)
            if (
                int(estimate.missed_count) == 0
                and int(estimate.hit_count) >= self.minimum_track_hits
                and self.minimum_speed_mps <= speed <= self.maximum_speed_mps
            ):
                self.samples.append((stamp, speed))
        if len(self.samples) < self.minimum_samples:
            self.last_prior_mps = None
            return rows
        prior = float(np.median([speed for _, speed in self.samples]))
        self.last_prior_mps = prior
        output: list[TrackEstimate] = []
        for estimate in rows:
            speed = float(estimate.speed_mps)
            if speed < .5:
                output.append(estimate)
                continue
            calibrated_speed = (
                (1.0 - self.prior_weight) * speed + self.prior_weight * prior
            )
            scale = calibrated_speed / speed
            output.append(replace(
                estimate, vx=float(estimate.vx * scale), vy=float(estimate.vy * scale)
            ))
            self.calibrated_count += 1
        return tuple(output)

    def status(self) -> dict[str, int | float | None]:
        return {
            "history_window_s": self.history_window_s,
            "prior_weight": self.prior_weight,
            "minimum_samples": self.minimum_samples,
            "minimum_track_hits": self.minimum_track_hits,
            "active_samples": len(self.samples),
            "last_prior_mps": self.last_prior_mps,
            "calibrated": self.calibrated_count,
        }


@dataclass
class _SemanticObject:
    object_id: int
    birth_stamp_ns: int
    last_measurement_stamp_ns: int
    estimate: TrackEstimate
    semantic_probability: float
    missed_updates: int = 0


class SemanticObjectMemory:
    """只让高置信轨迹出生，并用连续运动承接后续低置信片段。"""

    def __init__(
        self,
        *,
        birth_probability: float = 0.90,
        association_position_gate_m: float = 2.0,
        association_velocity_gate_mps: float = 0.7,
        maximum_memory_age_s: float = 5.0,
        publication_coast_s: float = 0.7,
        spawn_separation_m: float = 2.0,
        velocity_previous_weight: float = 0.5,
        minimum_birth_evidence_hits: int = 3,
        birth_evidence_max_gap_s: float = 0.35,
        semantic_probability_previous_weight: float = .9,
        semantic_sustain_probability: float = 0.0,
        semantic_sustain_grace_s: float = 2.0,
    ) -> None:
        self.birth_probability = float(birth_probability)
        self.association_position_gate_m = float(association_position_gate_m)
        self.association_velocity_gate_mps = float(association_velocity_gate_mps)
        self.maximum_memory_age_s = float(maximum_memory_age_s)
        self.publication_coast_s = float(publication_coast_s)
        self.spawn_separation_m = float(spawn_separation_m)
        self.velocity_previous_weight = float(velocity_previous_weight)
        self.minimum_birth_evidence_hits = int(minimum_birth_evidence_hits)
        self.birth_evidence_max_gap_s = float(birth_evidence_max_gap_s)
        self.semantic_probability_previous_weight = float(semantic_probability_previous_weight)
        self.semantic_sustain_probability = float(semantic_sustain_probability)
        self.semantic_sustain_grace_s = float(semantic_sustain_grace_s)
        if not (0.0 <= self.birth_probability <= 1.0):
            raise ValueError("SEMANTIC_BIRTH_PROBABILITY_INVALID")
        if min(
            self.association_position_gate_m, self.association_velocity_gate_mps,
            self.maximum_memory_age_s, self.publication_coast_s, self.spawn_separation_m,
        ) <= 0.0 or not (0.0 <= self.velocity_previous_weight <= 1.0):
            raise ValueError("SEMANTIC_OBJECT_MEMORY_PARAMETER_INVALID")
        if self.minimum_birth_evidence_hits < 1 or self.birth_evidence_max_gap_s <= 0.0:
            raise ValueError("SEMANTIC_BIRTH_EVIDENCE_PARAMETER_INVALID")
        if not 0.0 <= self.semantic_probability_previous_weight <= 1.0:
            raise ValueError("SEMANTIC_PROBABILITY_WEIGHT_INVALID")
        if not 0.0 <= self.semantic_sustain_probability <= 1.0 or self.semantic_sustain_grace_s < 0.0:
            raise ValueError("SEMANTIC_SUSTAIN_PARAMETER_INVALID")
        self.objects: dict[int, _SemanticObject] = {}
        self.birth_evidence: dict[int, tuple[int, int]] = {}
        self.next_object_id = 1
        self.birth_count = 0
        self.expiration_count = 0
        self.match_count = 0
        self.low_probability_match_count = 0

    @staticmethod
    def _predict_xy(obj: _SemanticObject, stamp_ns: int) -> tuple[float, float]:
        dt = max(0.0, (int(stamp_ns) - obj.last_measurement_stamp_ns) / 1e9)
        return obj.estimate.x + obj.estimate.vx * dt, obj.estimate.y + obj.estimate.vy * dt

    def update(
        self,
        stamp_ns: int,
        scored_estimates: tuple[tuple[TrackEstimate, float], ...],
    ) -> tuple[TrackEstimate, ...]:
        # 底层外推没有新的雷达测量，不能刷新公共对象的测量时刻。
        # 对象仍按最后真实测量外推，当前真实观测回来后可继续关联原 ID。
        scored_estimates = tuple((estimate, probability) for estimate, probability in scored_estimates
                                 if int(estimate.missed_count) == 0)
        expired = [
            object_id for object_id, obj in self.objects.items()
            if (int(stamp_ns) - obj.last_measurement_stamp_ns) / 1e9 > self.maximum_memory_age_s
        ]
        for object_id in expired:
            del self.objects[object_id]
            self.expiration_count += 1

        object_ids = sorted(self.objects)
        costs: list[list[float | None]] = []
        for object_id in object_ids:
            obj = self.objects[object_id]
            predicted_x, predicted_y = self._predict_xy(obj, stamp_ns)
            row: list[float | None] = []
            for estimate, _ in scored_estimates:
                position_error = math.hypot(estimate.x - predicted_x, estimate.y - predicted_y)
                velocity_error = math.hypot(estimate.vx - obj.estimate.vx, estimate.vy - obj.estimate.vy)
                if position_error > self.association_position_gate_m or velocity_error > self.association_velocity_gate_mps:
                    row.append(None)
                else:
                    row.append(position_error / self.association_position_gate_m + velocity_error / self.association_velocity_gate_mps)
            costs.append(row)
        association = exact_one_to_one_assignment(
            costs, detection_count=len(scored_estimates), unmatched_cost=3.0
        )
        matched_estimates: set[int] = set()
        for object_index, estimate_index in association.matches:
            object_id = object_ids[object_index]
            obj = self.objects[object_id]
            estimate, probability = scored_estimates[estimate_index]
            weight = self.velocity_previous_weight
            obj.estimate = replace(
                estimate,
                track_id=object_id,
                vx=weight * obj.estimate.vx + (1.0 - weight) * estimate.vx,
                vy=weight * obj.estimate.vy + (1.0 - weight) * estimate.vy,
                classification="BOAT",
            )
            obj.last_measurement_stamp_ns = int(stamp_ns)
            if self.semantic_sustain_probability > 0.0:
                semantic_weight = self.semantic_probability_previous_weight
                obj.semantic_probability = (
                    semantic_weight * obj.semantic_probability
                    + (1.0 - semantic_weight) * float(probability)
                )
            else:
                obj.semantic_probability = max(obj.semantic_probability, float(probability))
            obj.missed_updates = 0
            matched_estimates.add(estimate_index)
            self.match_count += 1
            self.low_probability_match_count += int(probability < self.birth_probability)

        present_track_ids: set[int] = set()
        for estimate, probability in scored_estimates:
            source_track_id = int(estimate.track_id)
            present_track_ids.add(source_track_id)
            if probability < self.birth_probability or int(estimate.missed_count) != 0:
                continue
            previous_count, previous_stamp = self.birth_evidence.get(
                source_track_id, (0, int(stamp_ns))
            )
            if (
                int(stamp_ns) - previous_stamp
                > int(self.birth_evidence_max_gap_s * 1e9)
            ):
                previous_count = 0
            self.birth_evidence[source_track_id] = (
                previous_count + 1, int(stamp_ns)
            )
        for source_track_id, (_, evidence_stamp) in tuple(self.birth_evidence.items()):
            if (
                source_track_id not in present_track_ids
                and int(stamp_ns) - evidence_stamp
                > int(self.birth_evidence_max_gap_s * 1e9)
            ):
                del self.birth_evidence[source_track_id]

        for estimate_index, (estimate, probability) in enumerate(scored_estimates):
            evidence_count = self.birth_evidence.get(int(estimate.track_id), (0, 0))[0]
            if (
                estimate_index in matched_estimates
                or probability < self.birth_probability
                or int(estimate.missed_count) != 0
                or evidence_count < self.minimum_birth_evidence_hits
            ):
                continue
            if any(
                math.hypot(estimate.x - self._predict_xy(obj, stamp_ns)[0], estimate.y - self._predict_xy(obj, stamp_ns)[1])
                < self.spawn_separation_m
                for obj in self.objects.values()
            ):
                continue
            object_id = self.next_object_id
            self.next_object_id += 1
            self.objects[object_id] = _SemanticObject(
                object_id=object_id,
                birth_stamp_ns=int(stamp_ns),
                last_measurement_stamp_ns=int(stamp_ns),
                estimate=replace(estimate, track_id=object_id, classification="BOAT"),
                semantic_probability=float(probability),
            )
            self.birth_count += 1
            self.birth_evidence.pop(int(estimate.track_id), None)

        output: list[TrackEstimate] = []
        for object_id in sorted(self.objects):
            obj = self.objects[object_id]
            elapsed = max(0.0, (int(stamp_ns) - obj.last_measurement_stamp_ns) / 1e9)
            if elapsed > self.publication_coast_s:
                continue
            object_age = max(0.0, (int(stamp_ns) - obj.birth_stamp_ns) / 1e9)
            if (
                object_age > self.semantic_sustain_grace_s
                and obj.semantic_probability < self.semantic_sustain_probability
            ):
                continue
            if elapsed > 0.0:
                obj.missed_updates += 1
            x, y = self._predict_xy(obj, stamp_ns)
            output.append(replace(
                obj.estimate,
                track_id=object_id,
                x=x,
                y=y,
                age_sec=object_age,
                missed_count=obj.missed_updates,
                lifecycle="CONFIRMED" if elapsed <= 1e-9 else "COASTING",
                classification="BOAT",
                confidence=max(obj.estimate.confidence, obj.semantic_probability),
            ))
        return tuple(output)

    def status(self) -> dict[str, int | float]:
        return {
            "active_objects": len(self.objects), "births": self.birth_count,
            "expirations": self.expiration_count, "matches": self.match_count,
            "low_probability_matches": self.low_probability_match_count,
            "birth_probability": self.birth_probability,
            "association_position_gate_m": self.association_position_gate_m,
            "association_velocity_gate_mps": self.association_velocity_gate_mps,
            "maximum_memory_age_s": self.maximum_memory_age_s,
            "publication_coast_s": self.publication_coast_s,
            "minimum_birth_evidence_hits": self.minimum_birth_evidence_hits,
            "birth_evidence_max_gap_s": self.birth_evidence_max_gap_s,
            "semantic_probability_previous_weight": self.semantic_probability_previous_weight,
            "semantic_sustain_probability": self.semantic_sustain_probability,
            "semantic_sustain_grace_s": self.semantic_sustain_grace_s,
        }
