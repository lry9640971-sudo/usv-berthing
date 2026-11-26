"""泊位撤销、真实入口姿态、终点漂移及能力状态的任务契约。"""

import math

import pytest

from competition_user.closed_loop.dynamics import VesselState
from competition_user.closed_loop.mission import Mission, MissionState, Bay
from phase17_berthing_perception.berthing_core import make_berth_candidates


def payload(stamp, occupancy=("FREE", "FREE", "FREE"), status="READY"):
    rows = [{"status": item, "confidence": 1., "cells": 100} for item in occupancy]
    return {"frame_id": "team_enu", "stamp_ns": int(stamp*1e9), "status": status,
            "dock_pose_xyyaw_m_rad": [0., 0., 0.],
            "candidates": make_berth_candidates((0., 0., 0.), rows, valid=True,
                                                confidence=1., covariance=[0.]*9, reason="GEOMETRY")}


def test_occupied_bay_revokes_all_previous_goal_coordinates():
    mission = Mission()
    vessel = VesselState(-8., 3., 0., 0., 0., 0.)
    mission.accept_bays(payload(1.), 1.)
    mission.update(vessel, 1.)
    assert mission.bay.bay_id == 1 and mission.state == MissionState.FINAL
    mission.accept_bays(payload(1.1, ("OCCUPIED", "FREE", "UNKNOWN")), 1.1)
    assert mission.bay.bay_id == 1 and mission.pending_occupancy
    assert mission.bay_stamp == 1.
    mission.accept_bays(payload(1.2, ("OCCUPIED", "FREE", "UNKNOWN")), 1.2)
    assert mission.bay is None and mission.state == MissionState.NAVIGATE
    mission.update(vessel, 1.2)
    assert mission.bay.bay_id == 2 and mission.state == MissionState.APPROACH
    assert mission.guidance(vessel).goal_xy == (-8., 9.)


def test_expired_free_bay_cannot_reactivate_without_new_measurement():
    mission = Mission()
    vessel = VesselState(-8., 3., 0., 0., 0., 0.)
    mission.accept_bays(payload(1.), 1.)
    mission.update(vessel, 1.)
    mission.update(vessel, 4.1)
    assert mission.bay is None
    mission.update(vessel, 4.2)
    assert mission.bay is None
    mission.accept_bays(payload(4.2, status="DEGRADED"), 4.2)
    mission.update(vessel, 4.2)
    assert mission.bay is None


def test_completion_requires_observed_position_speed_and_current_capability():
    mission = Mission()
    vessel = VesselState(-8., 3., 0., 0., 0., 0.)
    mission.accept_bays(payload(1.), 1.)
    mission.update(vessel, 1.)
    at_goal = VesselState(1.5, 3., 0., .05, .03, .01)
    mission.update(at_goal, 1.2, can_complete=False)
    mission.update(at_goal, 2.5, can_complete=False)
    assert mission.state != MissionState.COMPLETE
    mission.update(at_goal, 2.5)
    mission.update(at_goal, 3.6)
    assert mission.state == MissionState.COMPLETE


def test_cross_error_keeps_one_terminal_goal_and_controller():
    mission = Mission()
    vessel = VesselState(-8., 3., 0., 0., 0., 0.)
    mission.accept_bays(payload(1.), 1.)
    mission.update(vessel, 1.)
    lateral = VesselState(1.2, 3.3, 0., .05, .03, 0.)
    first = mission.guidance(lateral)
    assert first.feedback == "terminal"
    # 横偏在旧门限两侧变化，不应把终点切成九米外的入口或另一个控制器。
    second = mission.guidance(VesselState(.3, 3.2, 0., -.1, 0., 0.))
    assert second.goal_xy == first.goal_xy == mission.bay.final_pose[:2]
    assert second.feedback == "terminal"
    third = mission.guidance(VesselState(-8., 3.05, 0., 0., 0., 0.))
    assert mission.state == MissionState.FINAL
    assert third.feedback == "terminal" and third.goal_xy == first.goal_xy


def test_lateral_error_inside_entrance_stays_in_force_optimizer():
    mission = Mission()
    mission.accept_bays(payload(1.), 1.)
    mission.update(VesselState(-8., 3., 0., 0., 0., 0.), 1.)
    guidance = mission.guidance(VesselState(-3., 3.35, .1, .1, 0., 0.))
    assert guidance.goal_xy == (1.5, 3.)
    assert guidance.feedback == "terminal"


def test_at_rotated_entrance_alignment_is_not_replaced_by_zero_length_bearing():
    mission = Mission()
    mission.bay = Bay(1, (-8., 3., 1.2), (1.5, 3., 1.2))
    mission.state = MissionState.APPROACH
    vessel = VesselState(-8., 3., .6, 0., 0., 0.)
    guidance = mission.guidance(vessel)
    assert not guidance.use_route
    assert guidance.desired_yaw_rate == pytest.approx(.48)


def test_cross_error_outside_turning_envelope_is_corrected_without_return_loop():
    mission = Mission()
    mission.accept_bays(payload(1.), 1.)
    mission.update(VesselState(-8., 3., 0., 0., 0., 0.), 1.)
    guidance = mission.guidance(VesselState(-7., 3.35, 0., .2, .05, 0.))
    assert guidance.goal_xy == (1.5, 3.)
    assert guidance.desired_yaw_rate < 0.


def test_backward_excursion_does_not_reactivate_old_entry_recovery():
    mission = Mission()
    mission.accept_bays(payload(1.), 1.)
    mission.update(VesselState(-8., 3., 0., 0., 0., 0.), 1.)
    mission.guidance(VesselState(1.2, 3.35, 0., 0., 0., 0.))
    guidance = mission.guidance(VesselState(-8.5, 3.35, 0., 0., 0., 0.))
    assert guidance.feedback == "terminal" and guidance.goal_xy == (1.5, 3.)


def test_official_completion_speed_does_not_require_extra_angular_gate():
    mission=Mission()
    mission.bay=Bay(1,(-8.,3.,0.),(1.5,3.,0.))
    mission.state=MissionState.FINAL
    # 位置8厘米、船速.18满足官方门；安全性仍由同周期完整轨迹检查提供。
    vessel=VesselState(1.5,3.08,.12,.18,0.,.13)
    mission.check_completion(vessel,1.,True)
    assert mission.state==MissionState.COMPLETE


def test_official_speed_or_current_safety_failure_cannot_complete():
    for speed,capable in ((.201,True),(.18,False)):
        mission=Mission()
        mission.bay=Bay(1,(-8.,3.,0.),(1.5,3.,0.))
        mission.state=MissionState.FINAL
        vessel=VesselState(1.5,3.08,0.,speed,0.,0.)
        mission.check_completion(vessel,1.,capable)
        mission.check_completion(vessel,2.1,capable)
        assert mission.state==MissionState.FINAL
