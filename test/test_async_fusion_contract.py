"""异步融合保留逐观测时间、支持权重和同源分离约束。"""

import math

import pytest

from phase16z_s3_runtime.s3_async_fusion import fuse_common_frame_candidates
from phase16z_s3_runtime.s3_candidate_repair import CandidateObservation


def observation(name, source, x, support, *, stamp=1_000_000_000, y=0.0):
    return CandidateObservation(
        stamp_ns=stamp, observation_id=name, x_m=x, y_m=y, radius_m=0.5,
        support=support, source_sensors=(source,), position_std_m=0.2,
        range_m=20.0, confidence=0.7, z_min_m=0.2, z_max_m=1.2,
        vertical_extent_m=1.0, entity_evidence=True,
    )


def test_next_observation_uses_updated_weighted_center():
    # 第三路只能匹配已经融合前两路后的中心，不能沿用最初的单路中心。
    rows = [observation("a", "front", 0.0, 100),
            observation("b", "left", 0.8, 99),
            observation("c", "back", 1.25, 98)]
    result = fuse_common_frame_candidates(rows, fusion_stamp_ns=1_000_000_000)
    assert len(result) == 1
    assert result[0].support == 297
    assert result[0].x_m == pytest.approx((0.8 * 99 + 1.25 * 98) / 297)
    assert result[0].source_sensors == ("back", "front", "left")


def test_nearby_same_sensor_objects_remain_separate():
    rows = [observation("boat1-front", "front", 0.0, 100),
            observation("boat2-front", "front", 0.7, 90),
            observation("boat2-left", "left", 0.72, 80)]
    result = fuse_common_frame_candidates(rows, fusion_stamp_ns=1_000_000_000)
    assert len(result) == 2
    assert result[0].source_sensors == ("front",)
    assert result[1].support == 170
    assert result[1].observation_id == "async-fused:boat2-front+boat2-left"


def test_original_measurement_times_and_motion_uncertainty_are_retained():
    first = observation("old", "front", 0.0, 20, stamp=900_000_000)
    second = observation("new", "left", 0.0, 10)
    result = fuse_common_frame_candidates([first, second], fusion_stamp_ns=1_000_000_000)
    assert len(result) == 1
    assert result[0].stamp_ns == 1_000_000_000
    assert result[0].observation_stamps_ns == (900_000_000, 1_000_000_000)
    assert result[0].position_std_m == pytest.approx(math.hypot(0.2, 0.3))
    assert result[0].entity_evidence


def test_overlong_time_span_is_rejected():
    with pytest.raises(ValueError, match="ASYNC_FUSION_TIME_SPAN_TOO_LARGE"):
        fuse_common_frame_candidates(
            [observation("old", "front", 0.0, 20, stamp=800_000_000),
             observation("new", "left", 0.0, 10)], fusion_stamp_ns=1_000_000_000,
        )
