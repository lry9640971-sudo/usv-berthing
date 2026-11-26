"""S3 候选检测修复核心。

本模块只负责合法传感器数据到共同 base frame 的候选提取和多路去重，
不读取 GT，也不把静态场景标签注入生产链。与旧实现相比，关键改变是：

* 不再用固定 ``distance < 2 m`` 代替本船几何过滤；
* 使用真实的 base_link 碰撞多边形联合体逐点过滤本船回波；
* 候选保留 support、来源、尺寸和位置不确定度；
* 同步组内只合并不同传感器的近邻候选，邻近船不会因单一 source 的
  排序而无条件吞并。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .s3_timestamped_frames import RigidTransform2D, sensor_to_base_xy


@dataclass(frozen=True)
class BaseScanPoint:
    """已经使用固定 sensor->base 外参转换的激光点。"""

    stamp_ns: int
    source_sensor: str
    beam_index: int
    range_m: float
    x_m: float
    y_m: float


@dataclass(frozen=True)
class CandidateObservation:
    """供定位和 tracker 使用的带证据候选。"""

    stamp_ns: int
    observation_id: str
    x_m: float
    y_m: float
    radius_m: float
    support: int
    source_sensors: tuple[str, ...]
    position_std_m: float
    range_m: float
    confidence: float
    source_beam_span: tuple[int, int] = (0, 0)
    observation_stamps_ns: tuple[int, ...] = ()
    z_m: float = 0.0
    z_min_m: float = 0.0
    z_max_m: float = 0.0
    vertical_extent_m: float = 0.0
    horizontal_major_extent_m: float = 0.0
    horizontal_minor_extent_m: float = 0.0
    surface_normal_up_abs: float = 1.0
    vertical_voxel_layers: int = 1
    vertical_surface_fraction: float = 0.0
    organized_vertical_stack_fraction: float = 0.0
    mean_min_vertical_range_delta_m: float = 5.0
    organized_normal_valid_fraction: float = 0.0
    entity_evidence: bool = False

    def __post_init__(self) -> None:
        if int(self.stamp_ns) <= 0:
            raise ValueError("CANDIDATE_STAMP_INVALID")
        if int(self.support) <= 0:
            raise ValueError("CANDIDATE_SUPPORT_INVALID")
        if not self.source_sensors:
            raise ValueError("CANDIDATE_SOURCE_MISSING")
        numeric_values = (
            self.x_m,
            self.y_m,
            self.radius_m,
            self.position_std_m,
            self.range_m,
            self.confidence,
            self.z_m,
            self.z_min_m,
            self.z_max_m,
            self.vertical_extent_m,
            self.horizontal_major_extent_m,
            self.horizontal_minor_extent_m,
            self.surface_normal_up_abs,
            self.vertical_surface_fraction,
            self.organized_vertical_stack_fraction,
            self.mean_min_vertical_range_delta_m,
            self.organized_normal_valid_fraction,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("CANDIDATE_VALUE_NONFINITE")
        if float(self.radius_m) <= 0.0 or float(self.position_std_m) <= 0.0:
            raise ValueError("CANDIDATE_GEOMETRY_INVALID")
        if not 0.0 < float(self.confidence) < 1.0:
            raise ValueError("CANDIDATE_CONFIDENCE_MUST_BE_PROBABILISTIC")
        if (
            float(self.z_max_m) < float(self.z_min_m)
            or float(self.vertical_extent_m) < 0.0
            or float(self.horizontal_major_extent_m) < 0.0
            or float(self.horizontal_minor_extent_m) < 0.0
            or not 0.0 <= float(self.surface_normal_up_abs) <= 1.0
            or not 0.0 <= float(self.vertical_surface_fraction) <= 1.0
            or not 0.0 <= float(self.organized_vertical_stack_fraction) <= 1.0
            or float(self.mean_min_vertical_range_delta_m) < 0.0
            or not 0.0 <= float(self.organized_normal_valid_fraction) <= 1.0
            or int(self.vertical_voxel_layers) < 1
        ):
            raise ValueError("CANDIDATE_SHAPE_INVALID")


@dataclass
class CandidateFunnel:
    """候选阶段可与外部五路漏斗对账的计数。"""

    input_points: int = 0
    accepted_points: int = 0
    dropped_nonfinite: int = 0
    dropped_sensor_range: int = 0
    dropped_self_body: int = 0
    transformed_points: int = 0
    clusters_seen: int = 0
    clusters_rejected_support: int = 0
    candidates_emitted: int = 0
    fused_candidates_emitted: int = 0
    drop_reasons: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_points": int(self.input_points),
            "accepted_points": int(self.accepted_points),
            "transformed_points": int(self.transformed_points),
            "clusters_seen": int(self.clusters_seen),
            "candidates_emitted": int(self.candidates_emitted),
            "fused_candidates_emitted": int(self.fused_candidates_emitted),
            "dropped": {
                "nonfinite": int(self.dropped_nonfinite),
                "sensor_range": int(self.dropped_sensor_range),
                "self_body": int(self.dropped_self_body),
                "cluster_support": int(self.clusters_rejected_support),
                "reasons": dict(sorted(self.drop_reasons.items())),
            },
        }


class FootprintUnion:
    """base_link 碰撞多边形联合体的无依赖点包含判断。"""

    def __init__(self, components: Sequence[Sequence[tuple[float, float]]], *, source: str = "inline") -> None:
        cleaned: list[tuple[tuple[float, float], ...]] = []
        for component in components:
            polygon = tuple((float(x), float(y)) for x, y in component)
            if len(polygon) >= 3:
                cleaned.append(polygon)
        if not cleaned:
            raise ValueError("SELF_FOOTPRINT_EMPTY")
        self.components = tuple(cleaned)
        self.source = str(source)
        # 这里只做 broad-phase 粗筛，最终仍执行原始多边形点包含判断。
        self.component_bounds = tuple(
            (
                min(point[0] for point in polygon) - 1e-9,
                max(point[0] for point in polygon) + 1e-9,
                min(point[1] for point in polygon) - 1e-9,
                max(point[1] for point in polygon) + 1e-9,
            )
            for polygon in self.components
        )

    @property
    def component_count(self) -> int:
        return len(self.components)

    @staticmethod
    def _point_on_segment(x: float, y: float, ax: float, ay: float, bx: float, by: float) -> bool:
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) > 1e-9:
            return False
        return min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9 and min(ay, by) - 1e-9 <= y <= max(ay, by) + 1e-9

    @classmethod
    def _point_in_polygon(cls, point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
        x, y = point
        inside = False
        previous_x, previous_y = polygon[-1]
        for current_x, current_y in polygon:
            if cls._point_on_segment(x, y, previous_x, previous_y, current_x, current_y):
                return True
            crosses = (current_y > y) != (previous_y > y)
            if crosses:
                intersection_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
                if x < intersection_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside

    def contains(self, x_m: float, y_m: float) -> bool:
        x, y = float(x_m), float(y_m)
        return any(
            left <= x <= right
            and bottom <= y <= top
            and self._point_in_polygon((x, y), polygon)
            for polygon, (left, right, bottom, top) in zip(self.components, self.component_bounds)
        )


def load_footprint_yaml(path: str | Path, *, expected_components: int | None = None) -> FootprintUnion:
    """读取封存的真实 base_link 多边形，不接受缺失或缩减配置。"""

    footprint_path = Path(path)
    if not footprint_path.exists():
        raise FileNotFoundError(f"SELF_FOOTPRINT_NOT_FOUND:{footprint_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("FOOTPRINT_LOADER_REQUIRES_PYYAML") from exc
    document = yaml.safe_load(footprint_path.read_text(encoding="utf-8"))
    rows = document.get("components_polygon_xy") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ValueError("SELF_FOOTPRINT_COMPONENTS_MISSING")
    components = []
    for row in rows:
        points = row.get("polygon_xy") if isinstance(row, dict) else None
        if not isinstance(points, list):
            raise ValueError("SELF_FOOTPRINT_COMPONENT_POLYGON_MISSING")
        components.append([(float(point[0]), float(point[1])) for point in points])
    if expected_components is not None and len(components) != int(expected_components):
        raise ValueError(f"SELF_FOOTPRINT_COMPONENT_COUNT:{len(components)}!=expected:{expected_components}")
    if "component_count" in document and len(components) != int(document["component_count"]):
        raise ValueError("SELF_FOOTPRINT_DECLARED_COMPONENT_COUNT_INVALID")
    return FootprintUnion(components, source=str(footprint_path))


def scan_to_base_points(
    *,
    stamp_ns: int,
    source_sensor: str,
    ranges_m: Sequence[float],
    angle_min_rad: float,
    angle_increment_rad: float,
    range_min_m: float,
    range_max_m: float,
    sensor_to_base: RigidTransform2D,
    self_footprint: FootprintUnion,
    funnel: CandidateFunnel | None = None,
) -> tuple[BaseScanPoint, ...]:
    """按 LaserScan 顺序转换并过滤五路点，保持 source 和 beam 元数据。"""

    if int(stamp_ns) <= 0:
        raise ValueError("SCAN_STAMP_INVALID")
    if not source_sensor:
        raise ValueError("SCAN_SENSOR_NAME_MISSING")
    result: list[BaseScanPoint] = []
    for beam_index, raw_range in enumerate(ranges_m):
        if funnel is not None:
            funnel.input_points += 1
        distance = float(raw_range)
        if not math.isfinite(distance):
            if funnel is not None:
                funnel.dropped_nonfinite += 1
                funnel.drop_reasons["nonfinite_range"] += 1
            continue
        if distance < float(range_min_m) or distance > float(range_max_m):
            if funnel is not None:
                funnel.dropped_sensor_range += 1
                funnel.drop_reasons["sensor_range"] += 1
            continue
        angle = float(angle_min_rad) + int(beam_index) * float(angle_increment_rad)
        (x_base, y_base), = sensor_to_base_xy(
            ((distance * math.cos(angle), distance * math.sin(angle)),),
            sensor_to_base,
        )
        # 使用与当前运行时 WAM-V 绑定的 25 个 base_link 碰撞组件过滤本船回波。
        # 过滤发生在 sensor->base 外参转换之后，不能用距离阈值替代真实几何。
        if self_footprint.contains(x_base, y_base):
            if funnel is not None:
                funnel.dropped_self_body += 1
                funnel.drop_reasons["true_ownship_footprint"] += 1
            continue
        result.append(
            BaseScanPoint(
                stamp_ns=int(stamp_ns),
                source_sensor=str(source_sensor),
                beam_index=int(beam_index),
                range_m=distance,
                x_m=x_base,
                y_m=y_base,
            )
        )
        if funnel is not None:
            funnel.accepted_points += 1
            funnel.transformed_points += 1
    return tuple(result)


def _candidate_confidence(support: int, position_std_m: float) -> float:
    support_term = float(support) / (float(support) + 3.0)
    uncertainty_term = math.exp(-max(0.0, float(position_std_m)) / 2.0)
    return min(0.95, max(0.05, support_term * uncertainty_term))


def cluster_base_scan(
    points: Sequence[BaseScanPoint],
    *,
    minimum_support: int = 2,
    range_jump_m: float = 0.75,
    point_gap_m: float = 1.25,
    funnel: CandidateFunnel | None = None,
) -> tuple[CandidateObservation, ...]:
    """在单路 scan 内按 beam 连续性聚类，且不丢失不确定度和来源。"""

    if not points:
        return ()
    ordered = sorted(points, key=lambda point: int(point.beam_index))
    clusters: list[list[BaseScanPoint]] = []
    current: list[BaseScanPoint] = []
    previous: BaseScanPoint | None = None
    for point in ordered:
        split = False
        if previous is not None:
            split = (
                int(point.beam_index) != int(previous.beam_index) + 1
                or abs(float(point.range_m) - float(previous.range_m)) > float(range_jump_m)
                or math.hypot(float(point.x_m) - float(previous.x_m), float(point.y_m) - float(previous.y_m)) > float(point_gap_m)
            )
        if split and current:
            clusters.append(current)
            current = []
        current.append(point)
        previous = point
    if current:
        clusters.append(current)
    if funnel is not None:
        funnel.clusters_seen += len(clusters)

    result: list[CandidateObservation] = []
    for cluster_index, cluster in enumerate(clusters):
        support = len(cluster)
        if support < int(minimum_support):
            if funnel is not None:
                funnel.clusters_rejected_support += 1
                funnel.drop_reasons["cluster_too_few_points"] += 1
            continue
        weights = [float(max(1.0, point.range_m)) for point in cluster]
        weight_sum = sum(weights)
        center_x = sum(weight * float(point.x_m) for weight, point in zip(weights, cluster)) / weight_sum
        center_y = sum(weight * float(point.y_m) for weight, point in zip(weights, cluster)) / weight_sum
        radial_deviations = [math.hypot(float(point.x_m) - center_x, float(point.y_m) - center_y) for point in cluster]
        radius = max(0.20, max(radial_deviations, default=0.20))
        spread = math.sqrt(sum(value * value for value in radial_deviations) / max(1, support))
        position_std = max(0.08, spread / math.sqrt(max(1, support)))
        range_mean = sum(float(point.range_m) for point in cluster) / float(support)
        source = str(cluster[0].source_sensor)
        first_beam = int(cluster[0].beam_index)
        last_beam = int(cluster[-1].beam_index)
        result.append(
            CandidateObservation(
                stamp_ns=int(cluster[0].stamp_ns),
                observation_id=f"{source}:{cluster[0].stamp_ns}:{first_beam}-{last_beam}:{cluster_index}",
                x_m=center_x,
                y_m=center_y,
                radius_m=radius,
                support=support,
                source_sensors=(source,),
                position_std_m=position_std,
                range_m=range_mean,
                confidence=_candidate_confidence(support, position_std),
                source_beam_span=(first_beam, last_beam),
                observation_stamps_ns=(int(cluster[0].stamp_ns),),
            )
        )
    if funnel is not None:
        funnel.candidates_emitted += len(result)
    return tuple(result)


def _fuse_gate_m(first: CandidateObservation, second: CandidateObservation) -> float:
    uncertainty = 2.5 * math.hypot(float(first.position_std_m), float(second.position_std_m))
    geometry = 0.20 * min(float(first.radius_m), float(second.radius_m))
    return max(0.45, min(1.20, uncertainty + geometry + 0.25))


def _fuse_group(group: Sequence[CandidateObservation]) -> CandidateObservation:
    if not group:
        raise ValueError("FUSE_EMPTY_GROUP")
    if len({int(item.stamp_ns) for item in group}) != 1:
        raise ValueError("FUSION_REQUIRES_COMMON_OBSERVATION_STAMP")
    total_support = sum(int(item.support) for item in group)
    center_x = sum(float(item.x_m) * int(item.support) for item in group) / float(total_support)
    center_y = sum(float(item.y_m) * int(item.support) for item in group) / float(total_support)
    scatter = sum(
        int(item.support) * ((float(item.x_m) - center_x) ** 2 + (float(item.y_m) - center_y) ** 2)
        for item in group
    ) / float(total_support)
    standard_error = math.sqrt(max(0.0, scatter) / max(1, total_support))
    position_std = max(0.06, min(2.0, math.hypot(standard_error, max(float(item.position_std_m) for item in group))))
    sources = tuple(sorted({source for item in group for source in item.source_sensors}))
    beam_start = min(item.source_beam_span[0] for item in group)
    beam_end = max(item.source_beam_span[1] for item in group)
    confidence = 1.0 - math.prod(1.0 - float(item.confidence) for item in group)
    confidence = min(0.97, max(0.05, confidence * math.exp(-position_std / 5.0)))
    return CandidateObservation(
        stamp_ns=int(group[0].stamp_ns),
        observation_id="fused:" + "+".join(sorted(item.observation_id for item in group)),
        x_m=center_x,
        y_m=center_y,
        z_m=sum(float(item.z_m) * int(item.support) for item in group) / float(total_support),
        z_min_m=min(float(item.z_min_m) for item in group),
        z_max_m=max(float(item.z_max_m) for item in group),
        vertical_extent_m=max(float(item.z_max_m) for item in group) - min(float(item.z_min_m) for item in group),
        horizontal_major_extent_m=max(float(item.horizontal_major_extent_m) for item in group),
        horizontal_minor_extent_m=max(float(item.horizontal_minor_extent_m) for item in group),
        surface_normal_up_abs=sum(float(item.surface_normal_up_abs) * int(item.support) for item in group) / float(total_support),
        vertical_voxel_layers=max(int(item.vertical_voxel_layers) for item in group),
        vertical_surface_fraction=sum(float(item.vertical_surface_fraction) * int(item.support) for item in group) / float(total_support),
        organized_vertical_stack_fraction=sum(float(item.organized_vertical_stack_fraction) * int(item.support) for item in group) / float(total_support),
        mean_min_vertical_range_delta_m=sum(float(item.mean_min_vertical_range_delta_m) * int(item.support) for item in group) / float(total_support),
        organized_normal_valid_fraction=sum(float(item.organized_normal_valid_fraction) * int(item.support) for item in group) / float(total_support),
        entity_evidence=any(bool(item.entity_evidence) for item in group),
        radius_m=max(0.20, max(float(item.radius_m) for item in group)),
        support=total_support,
        source_sensors=sources,
        position_std_m=position_std,
        range_m=sum(float(item.range_m) * int(item.support) for item in group) / float(total_support),
        confidence=confidence,
        source_beam_span=(beam_start, beam_end),
        observation_stamps_ns=tuple(sorted({int(stamp) for item in group for stamp in (item.observation_stamps_ns or (item.stamp_ns,))})),
    )


def fuse_synchronized_candidates(
    candidates: Iterable[CandidateObservation],
    *,
    funnel: CandidateFunnel | None = None,
) -> tuple[CandidateObservation, ...]:
    """同步组内跨 sensor 去重，保留相邻独立目标。"""

    groups: list[list[CandidateObservation]] = []
    candidates = tuple(candidates)
    if len({int(item.stamp_ns) for item in candidates}) > 1:
        raise ValueError("FUSION_REQUIRES_COMMON_OBSERVATION_STAMP")
    ordered = sorted(candidates, key=lambda item: (-int(item.support), item.x_m, item.y_m, item.observation_id))
    for candidate in ordered:
        choices: list[tuple[float, int]] = []
        for group_index, group in enumerate(groups):
            if set(candidate.source_sensors) & {source for item in group for source in item.source_sensors}:
                continue
            representative = _fuse_group(group)
            distance = math.hypot(float(candidate.x_m) - representative.x_m, float(candidate.y_m) - representative.y_m)
            if distance <= _fuse_gate_m(candidate, representative):
                choices.append((distance, group_index))
        if choices:
            _, selected = min(choices)
            groups[selected].append(candidate)
        else:
            groups.append([candidate])
    fused = tuple(_fuse_group(group) for group in groups)
    if funnel is not None:
        funnel.fused_candidates_emitted += len(fused)
    return fused


__all__ = [
    "BaseScanPoint",
    "CandidateFunnel",
    "CandidateObservation",
    "FootprintUnion",
    "cluster_base_scan",
    "fuse_synchronized_candidates",
    "load_footprint_yaml",
    "scan_to_base_points",
]
