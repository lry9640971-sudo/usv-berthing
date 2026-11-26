"""低频稀疏点云热修复的纯逻辑回归。"""

from collections import deque
from dataclasses import dataclass

from phase16z_s3_runtime.s3_partial_sync import pop_ready_group
from phase16z_s3_runtime.s3_track_semantics import motion_supported_semantic_probability


@dataclass(frozen=True)
class _Row:
    topic: str
    stamp_ns: int


def test_full_group_is_emitted_without_waiting_for_next_frame() -> None:
    rows = {name: deque([_Row(name, 1_000_000_000)]) for name in "abcde"}
    selected = pop_ready_group(rows, tolerance_ns=100_000_000)
    assert selected is not None
    group, stamp, stale = selected
    assert len(group) == 5
    assert stamp == 1_000_000_000
    assert stale == 0


def test_slow_or_missing_sensor_does_not_block_fast_sensors() -> None:
    rows = {
        "a": deque([_Row("a", 1_000_000_000), _Row("a", 1_250_000_000)]),
        "b": deque([_Row("b", 1_020_000_000)]),
        "c": deque([_Row("c", 1_030_000_000)]),
        "d": deque([_Row("d", 1_040_000_000)]),
        "slow": deque(),
    }
    selected = pop_ready_group(rows, tolerance_ns=100_000_000)
    assert selected is not None
    group, stamp, _ = selected
    assert {row.topic for row in group} == {"a", "b", "c", "d"}
    assert stamp == 1_040_000_000


def test_low_support_motion_is_supported_but_static_jitter_is_not() -> None:
    moving = {
        "recent_observation_count": 6,
        "recent_motion_span_s": 1.25,
        "recent_motion_speed_mps": 0.8,
        "recent_path_efficiency": 0.94,
        "recent_line_fit_rms_m": 0.08,
    }
    probability, supported = motion_supported_semantic_probability(moving, 0.08)
    assert supported
    assert probability >= 0.75

    static = {
        "recent_observation_count": 8,
        "recent_motion_span_s": 1.5,
        "recent_motion_speed_mps": 0.12,
        "recent_path_efficiency": 0.90,
        "recent_line_fit_rms_m": 0.04,
    }
    probability, supported = motion_supported_semantic_probability(static, 0.08)
    assert not supported
    assert probability == 0.08


def test_close_collision_course_uses_stricter_fast_path() -> None:
    close_motion = {
        "track_distance_to_ego": 8.0,
        "recent_observation_count": 4,
        "recent_motion_span_s": 0.75,
        "recent_motion_speed_mps": 0.9,
        "recent_path_efficiency": 0.98,
        "recent_line_fit_rms_m": 0.05,
    }
    probability, supported = motion_supported_semantic_probability(close_motion, 0.05)
    assert supported
    assert probability >= 0.75


def test_motion_evidence_can_follow_a_stricter_birth_threshold() -> None:
    """提高语义门槛时，已确认的连续运动通路仍应能够出生。"""
    moving = {
        "track_distance_to_ego": 12.0,
        "recent_observation_count": 6,
        "recent_motion_span_s": 1.25,
        "recent_motion_speed_mps": 0.8,
        "recent_path_efficiency": 0.96,
        "recent_line_fit_rms_m": 0.06,
    }
    probability, supported = motion_supported_semantic_probability(
        moving, 0.08, supported_probability=0.90
    )
    assert supported
    assert probability >= 0.90


def test_close_self_attached_echo_cannot_birth_from_motion_alone() -> None:
    attached_echo = {
        "track_distance_to_ego": 3.6,
        "recent_observation_count": 8,
        "recent_motion_span_s": 1.5,
        "recent_motion_speed_mps": 1.0,
        "recent_path_efficiency": 0.98,
        "recent_line_fit_rms_m": 0.04,
    }
    probability, supported = motion_supported_semantic_probability(attached_echo, 0.08)
    assert not supported
    assert probability == 0.08
