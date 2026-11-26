"""公共对象不能把底层预测伪装成新的雷达测量。"""

from dataclasses import replace

import pytest

from phase16z_s3_runtime.s3_track_semantics import SemanticObjectMemory
from phase16z_s3_runtime.s3_tracker_repair import TrackEstimate


def estimate(stamp_s, missed=0):
    return TrackEstimate(track_id=8803, x=10.+stamp_s, y=2., vx=1., vy=0., radius_m=.6,
        confidence=.9, age_sec=stamp_s, covariance=(.01,)*16, hit_count=12, missed_count=missed,
        lifecycle="COASTING" if missed else "CONFIRMED", classification="BOAT",
        position_covariance=(.01,0.,0.,.01), velocity_covariance=(.02,0.,0.,.02), source_sensors=("back",))


def test_predicted_primary_tracks_do_not_refresh_public_measurement_time():
    memory = SemanticObjectMemory(minimum_birth_evidence_hits=1, publication_coast_s=.7)
    born = memory.update(0, ((estimate(0.),.95),))
    object_id = born[0].track_id
    for index, stamp in enumerate((.2,.4,.6), 1):
        output = memory.update(round(stamp*1e9), ((estimate(stamp,index),.95),))
        assert output[0].lifecycle == "COASTING"
        assert output[0].missed_count > 0
        assert output[0].x == pytest.approx(10.+stamp)
        assert memory.objects[object_id].last_measurement_stamp_ns == 0
    assert memory.update(800_000_000, ((estimate(.8,4),.95),)) == ()


def test_real_measurement_can_reacquire_same_object_after_public_coast_timeout():
    memory = SemanticObjectMemory(minimum_birth_evidence_hits=1, publication_coast_s=.7)
    object_id = memory.update(0, ((estimate(0.),.95),))[0].track_id
    assert memory.update(1_000_000_000, ((estimate(1.,5),.95),)) == ()
    measured = replace(estimate(1.2),track_id=9240)
    output = memory.update(1_200_000_000, ((measured,.65),))
    assert output[0].track_id == object_id
    assert output[0].lifecycle == "CONFIRMED"
    assert output[0].missed_count == 0
    assert memory.objects[object_id].last_measurement_stamp_ns == 1_200_000_000
