"""真实导航无解与两段保持力的安全、执行时序和持续运动反例。"""

from dataclasses import replace
import math
import time

import numpy as np
import pytest

from competition_user.closed_loop.dynamics import Disturbance, VesselState
from competition_user.closed_loop.force_sequence import TimedForceSequence, predict_sequence
from competition_user.closed_loop.geometry import moving_clearances
from competition_user.closed_loop.planner import Guidance
from test_navigation_force_certification import recorded


def test_real_no_safe_snapshot_has_certified_two_stage_thrust_escape():
    fixture, planner, field, moving = recorded(450800)
    assert fixture["old_plan_status"] == "NO_SAFE_TRAJECTORY"
    plan = planner.choose(VesselState(**fixture["vessel"]), Guidance(**fixture["guidance"]),
                          field, moving, Disturbance(**fixture["disturbance"]))
    assert plan.status == "SAFE"
    diagnostic = planner.last_diagnostics
    assert diagnostic["decision"] == "navigation_force_sequence_escape"
    assert 0 < diagnostic["sequence_candidates_evaluated"] <= len(planner.navigation_sequences.controls)
    assert len(diagnostic["selected_controls_n"]) == 2
    assert sum(diagnostic["selected_force_durations_s"]) == pytest.approx(4.)
    fine = predict_sequence(planner.model, plan.trajectory[0],
        np.asarray(diagnostic["selected_controls_n"])/100., diagnostic["selected_force_durations_s"],
        Disturbance(**fixture["disturbance"]), 200, .02)
    assert field.clearances(fine[:, :3]).min() > 0.
    assert moving_clearances(fine[:, :3], field.hull, moving, .02).min() > 0.
    assert diagnostic["swept_certification"]["static_lower_bound_m"] >= 0.


def test_escape_restarts_at_actual_execution_state_after_old_force_delay():
    fixture, planner, field, moving = recorded(450800)
    initial, disturbance = VesselState(**fixture["vessel"]), Disturbance(**fixture["disturbance"])
    old = planner.model.actuator.command(**fixture["previous_force_n"])
    current = planner.model.step(initial, old, .05, disturbance)
    advanced = [replace(item, x=item.x+item.vx*.05, y=item.y+item.vy*.05) for item in moving]
    plan = planner.choose(initial, Guidance(**fixture["guidance"]), field, moving, disturbance,
        execution_snapshot=lambda: (current, advanced, fixture["stamp_sim_s"]+.05))
    assert plan.status == "SAFE" and plan.trajectory[0] == current
    assert planner.last_diagnostics["prediction_stamp_sim_s"] == pytest.approx(fixture["stamp_sim_s"]+.05)
    assert moving_clearances([item.pose for item in plan.trajectory], field.hull, advanced, .2).min() >= 0.


def test_infeasible_sequence_never_overwrites_original_no_safe_recovery():
    fixture, planner, field, moving = recorded(452400)
    plan = planner.choose(VesselState(**fixture["vessel"]), Guidance(**fixture["guidance"]), field,
                          moving, Disturbance(**fixture["disturbance"]))
    assert plan.status == "NO_SAFE_TRAJECTORY" and plan.safe_candidate_count == 0
    assert plan.command.__dict__ == pytest.approx(fixture["old_published_force_n"])
    assert "selected_controls_n" not in planner.last_diagnostics


def test_elapsed_budget_does_not_approve_unexamined_sequences():
    fixture, planner, field, moving = recorded(450800)
    result = planner.navigation_sequences.choose(VesselState(**fixture["vessel"]),
        Guidance(**fixture["guidance"]), field, moving, Disturbance(**fixture["disturbance"]),
        deadline=time.perf_counter()-1.)
    assert result is None
    assert planner.navigation_sequences.last_diagnostics["sequence_budget_reached"]
    assert planner.navigation_sequences.last_diagnostics["sequence_candidates_evaluated"] == 0


def test_budget_expiring_during_geometry_stops_remaining_candidates_without_false_safe(monkeypatch):
    from competition_user.closed_loop import navigation_sequences

    fixture, planner, field, moving = recorded(450800)
    clock = [0.]
    original_clearances = field.clearances

    def slow_geometry(poses):
        result = original_clearances(poses)
        # 已完成当前几何调用，但预算耗尽；其余候选及精查不能继续或假定通过。
        clock[0] = 2.
        return result

    monkeypatch.setattr(field, "clearances", slow_geometry)
    monkeypatch.setattr(navigation_sequences.time, "perf_counter", lambda: clock[0])
    result = planner.navigation_sequences.choose(VesselState(**fixture["vessel"]),
        Guidance(**fixture["guidance"]), field, moving, Disturbance(**fixture["disturbance"]), deadline=1.)
    diagnostic = planner.navigation_sequences.last_diagnostics
    assert result is None and diagnostic["sequence_budget_reached"]
    assert 0 < diagnostic["sequence_candidates_evaluated"] < len(planner.navigation_sequences.controls)
    assert diagnostic["sequence_certification_attempts"] == 0


def test_actual_dense_navigation_failure_is_not_approved_from_coarse_samples():
    fixture, planner, field, moving = recorded(189304)
    state = VesselState(**fixture["prediction_initial_vessel"])
    stamp = fixture["prediction_stamp_sim_s"]
    assert fixture["old_plan_status"] == "NO_SAFE_TRAJECTORY" and len(field.points) == 7145
    plan = planner.choose(VesselState(**fixture["vessel"]), Guidance(**fixture["guidance"]),
        field, moving, Disturbance(**fixture["disturbance"]), execution_snapshot=lambda: (state, moving, stamp))
    assert plan.status == "NO_SAFE_TRAJECTORY" and plan.safe_candidate_count == 0
    assert plan.command.__dict__ == pytest.approx(fixture["old_published_force_n"])
    assert planner.last_diagnostics["selection_at_execution_state"]
    assert planner.last_diagnostics["prediction_stamp_sim_s"] == pytest.approx(stamp)


def test_published_force_sequence_and_delayed_replanning_remain_clear_for_three_seconds():
    fixture, planner, field, original_moving = recorded(450800)
    state, disturbance = VesselState(**fixture["vessel"]), Disturbance(**fixture["disturbance"])
    original_guidance = Guidance(**fixture["guidance"])
    old = planner.model.actuator.command(**fixture["previous_force_n"])
    schedule = TimedForceSequence([(old.left_n, old.right_n)], [4.], 0.)
    actual_static, actual_dynamic = [], []
    sequence_decisions = 0

    def advance(current, begin, end):
        for first, last, force in schedule.intervals(begin, end):
            count = math.ceil((last-first)/.02)
            for index in range(count):
                current = planner.model.step(current, force, (last-first)/count, disturbance)
                stamp = first+(index+1)*(last-first)/count
                moved = [replace(item, x=item.x+item.vx*stamp, y=item.y+item.vy*stamp) for item in original_moving]
                actual_static.append(float(field.clearances([current.pose])[0]))
                actual_dynamic.append(float(moving_clearances([current.pose], field.hull, moved, .02).min()))
        return current

    for index in range(25):
        elapsed = index*.12
        error = math.atan2(original_guidance.goal_xy[1]-state.y, original_guidance.goal_xy[0]-state.x)-state.yaw
        error = math.atan2(math.sin(error), math.cos(error))
        guidance = replace(original_guidance,
            desired_surge=.85*max(0., math.cos(error))**2,
            desired_yaw_rate=max(-.5, min(.5, .8*error)))
        current = advance(state, elapsed, elapsed+.05)
        moved = [replace(item, x=item.x+item.vx*elapsed, y=item.y+item.vy*elapsed) for item in original_moving]
        advanced = [replace(item, x=item.x+item.vx*.05, y=item.y+item.vy*.05) for item in moved]
        plan = planner.choose(state, guidance, field, moved, disturbance,
            execution_snapshot=lambda: (current, advanced, elapsed+.05))
        diagnostic = planner.last_diagnostics
        if plan.status == "SAFE" and "selected_controls_n" in diagnostic:
            sequence_decisions += 1
            schedule = TimedForceSequence(diagnostic["selected_controls_n"],
                diagnostic["selected_force_durations_s"], diagnostic["prediction_stamp_sim_s"])
        else:
            schedule = TimedForceSequence([(plan.command.left_n, plan.command.right_n)], [4.], elapsed+.05)
        state = advance(current, elapsed+.05, elapsed+.12)
    assert sequence_decisions > 0
    assert min(actual_static) >= 0. and min(actual_dynamic) >= 0.
