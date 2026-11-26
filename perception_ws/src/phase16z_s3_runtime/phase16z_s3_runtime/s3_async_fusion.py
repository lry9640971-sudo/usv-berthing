"""按观测时间先转公共坐标，再做异步多传感器融合。

旧 v1 的 ``fuse_synchronized_candidates`` 仍严格拒绝不同时间戳；本模块
是今晚 ROS 链使用的显式异步路径。每个候选先按自己的 stamp 用 GPS/IMU
pose 转到 team_enu，融合时不伪造原始观测时间，而是把时间跨度折算为
目标运动不确定度。自船运动对静态背景的影响已在逐观测 ENU 变换中消除。
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Iterable, Sequence

from .s3_candidate_repair import CandidateObservation


def _gate(first: CandidateObservation, second: CandidateObservation) -> float:
    uncertainty = 2.5 * math.hypot(float(first.position_std_m), float(second.position_std_m))
    geometry = 0.20 * min(float(first.radius_m), float(second.radius_m))
    return max(0.45, min(1.8, uncertainty + geometry + 0.25))


def _combine_common(group: Sequence[CandidateObservation], fusion_stamp_ns: int) -> CandidateObservation:
    if not group:
        raise ValueError("ASYNC_FUSION_EMPTY_GROUP")
    total_support = sum(int(item.support) for item in group)
    center_x = sum(float(item.x_m) * int(item.support) for item in group) / float(total_support)
    center_y = sum(float(item.y_m) * int(item.support) for item in group) / float(total_support)
    scatter = sum(
        int(item.support) * ((float(item.x_m) - center_x) ** 2 + (float(item.y_m) - center_y) ** 2)
        for item in group
    ) / float(total_support)
    standard_error = math.sqrt(max(0.0, scatter) / max(1, total_support))
    position_std = max(0.06, min(5.0, math.hypot(standard_error, max(float(item.position_std_m) for item in group))))
    sources = tuple(sorted({source for item in group for source in item.source_sensors}))
    confidence = 1.0 - math.prod(1.0 - float(item.confidence) for item in group)
    confidence = min(0.97, max(0.05, confidence * math.exp(-position_std / 5.0)))
    stamps = tuple(sorted({int(stamp) for item in group for stamp in item.observation_stamps_ns}))
    if not stamps:
        stamps = tuple(sorted({int(item.stamp_ns) for item in group}))
    return CandidateObservation(
        stamp_ns=int(fusion_stamp_ns),
        observation_id="async-fused:" + "+".join(sorted(item.observation_id for item in group)),
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
        source_beam_span=(min(item.source_beam_span[0] for item in group), max(item.source_beam_span[1] for item in group)),
        observation_stamps_ns=stamps,
    )


def fuse_common_frame_candidates(
    candidates: Iterable[CandidateObservation],
    *,
    fusion_stamp_ns: int,
    maximum_time_span_ns: int = 150_000_000,
    target_speed_bound_mps: float = 3.0,
) -> tuple[CandidateObservation, ...]:
    """在已转入同一公共坐标的候选上进行有界异步融合。"""

    rows = tuple(candidates)
    if not rows:
        return ()
    if int(fusion_stamp_ns) <= 0:
        raise ValueError("ASYNC_FUSION_STAMP_INVALID")
    if int(maximum_time_span_ns) <= 0 or float(target_speed_bound_mps) < 0.0:
        raise ValueError("ASYNC_FUSION_PARAMETER_INVALID")
    stamps = [int(item.stamp_ns) for item in rows]
    if max(stamps) - min(stamps) > int(maximum_time_span_ns):
        raise ValueError("ASYNC_FUSION_TIME_SPAN_TOO_LARGE")

    # 时间跨度只放大测量不确定度，不改写任何输入观测的来源时间。
    temporally_uncertain = tuple(
        replace(
            item,
            stamp_ns=int(fusion_stamp_ns),
            position_std_m=math.hypot(
                float(item.position_std_m),
                float(target_speed_bound_mps) * abs(int(item.stamp_ns) - int(fusion_stamp_ns)) / 1_000_000_000.0,
            ),
            observation_stamps_ns=tuple(sorted(set(item.observation_stamps_ns or (int(item.stamp_ns),)))),
        )
        for item in rows
    )
    groups: list[list[CandidateObservation]] = []
    representatives: list[CandidateObservation] = []
    group_sources: list[set[str]] = []
    ordered = sorted(temporally_uncertain, key=lambda item: (-int(item.support), item.x_m, item.y_m, item.observation_id))
    for candidate in ordered:
        choices: list[tuple[float, int]] = []
        candidate_sources = set(candidate.source_sensors)
        for group_index, representative in enumerate(representatives):
            if candidate_sources & group_sources[group_index]:
                continue
            distance = math.hypot(float(candidate.x_m) - representative.x_m, float(candidate.y_m) - representative.y_m)
            if distance <= _gate(candidate, representative):
                choices.append((distance, group_index))
        if choices:
            _, selected = min(choices)
            groups[selected].append(candidate)
            group_sources[selected].update(candidate_sources)
            # 组内容改变时才重算全部几何和证据，后续候选仍使用更新后的代表。
            representatives[selected] = _combine_common(groups[selected], int(fusion_stamp_ns))
        else:
            groups.append([candidate])
            group_sources.append(candidate_sources)
            representatives.append(_combine_common(groups[-1], int(fusion_stamp_ns)))
    return tuple(representatives)


__all__ = ["fuse_common_frame_candidates"]
