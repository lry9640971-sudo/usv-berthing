"""S3 派生候选、时间戳位姿和公开 DynamicTrack 的适配层。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from .s3_candidate_repair import CandidateObservation
from .s3_timestamped_frames import TimestampedPoseBuffer, base_to_enu_xy
from .s3_tracker_repair import TrackEstimate


def candidates_base_to_team_enu(
    candidates: Iterable[CandidateObservation],
    pose_buffer: TimestampedPoseBuffer,
) -> tuple[CandidateObservation, ...]:
    """使用每个候选观测时刻的 pose 转到 team_enu。"""

    result = []
    for candidate in candidates:
        pose = pose_buffer.interpolate(int(candidate.stamp_ns))
        x_enu, y_enu = base_to_enu_xy((candidate.x_m, candidate.y_m), pose)
        result.append(replace(candidate, x_m=x_enu, y_m=y_enu, observation_id=f"enu:{candidate.observation_id}"))
    return tuple(result)


def track_estimate_to_public_dict(estimate: TrackEstimate, *, frame_id: str, stamp_ns: int) -> dict[str, Any]:
    """映射到现有 DynamicTrack.msg 可表达的字段，并保留审计字段。"""

    position_covariance = estimate.position_covariance
    velocity_covariance = estimate.velocity_covariance
    return {
        "stamp_ns": int(stamp_ns),
        "frame_id": str(frame_id),
        "track_id": int(estimate.track_id),
        "x_m": float(estimate.x),
        "y_m": float(estimate.y),
        "vx_mps": float(estimate.vx),
        "vy_mps": float(estimate.vy),
        "speed_mps": float(estimate.speed_mps),
        "heading_rad": float(estimate.heading_rad),
        "radius_m": float(estimate.radius_m),
        "confidence": float(estimate.confidence),
        "position_covariance_xx": float(position_covariance[0]),
        "position_covariance_xy": float(position_covariance[1]),
        "position_covariance_yy": float(position_covariance[3]),
        "velocity_covariance_xx": float(velocity_covariance[0]),
        "velocity_covariance_xy": float(velocity_covariance[1]),
        "velocity_covariance_yy": float(velocity_covariance[3]),
        "age_sec": float(estimate.age_sec),
        "hit_count": int(estimate.hit_count),
        "missed_count": int(estimate.missed_count),
        "lifecycle": estimate.lifecycle,
        "classification": estimate.classification,
        "source_sensors": list(estimate.source_sensors),
        "gt_feedback": False,
    }


__all__ = ["candidates_base_to_team_enu", "track_estimate_to_public_dict"]
