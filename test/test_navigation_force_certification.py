"""真实碰撞前受力候选、采样间隙及执行时刻的安全反例。"""

from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import pytest

from competition_user.closed_loop.dynamics import Disturbance, VesselDynamics, VesselState
from competition_user.closed_loop.force_sequence import predict_sequence
from competition_user.closed_loop.geometry import CollisionField, Hull, certify_swept_poses, moving_clearances
from competition_user.closed_loop.planner import Guidance, MovingObstacle, ThrustPlanner

ROOT = Path(__file__).resolve().parents[1]


def tiny_hull():
    return Hull([(-.1, -.1), (.1, -.1), (.1, .1), (-.1, .1)])


def recorded(stamp=410104):
    fixture = json.loads((ROOT / f"test/fixtures/navigation_{stamp}.json").read_text(encoding="utf-8"))
    data = np.load(ROOT / f"test/fixtures/navigation_{stamp}.npz", allow_pickle=False)
    model = VesselDynamics.load(ROOT / "control_ws/src/competition_user/config/vessel_dynamics.yaml")
    hull = Hull.load(ROOT / "perception_ws/src/phase17_berthing_perception/config/hull_collision_footprint_base_link.yaml")
    moving = [MovingObstacle(**{key: value for key, value in item.items() if key != "track_id"})
              for item in fixture["moving_obstacles"]]
    field = CollisionField(data["points"], hull, float(data["point_radius_m"]))
    planner = ThrustPlanner(model, hull)
    planner.previous_setpoint = tuple(fixture["previous_setpoint"])
    return fixture, planner, field, moving


def test_swept_static_segment_rejects_collision_between_safe_endpoints():
    field = CollisionField([(0., 0.)], tiny_hull())
    poses = np.asarray([(-1., 0., 0.), (1., 0., 0.)])
    assert field.clearances(poses).min() > .8
    safe, static, _, diagnostic = certify_swept_poses(poses, field, [], [0., 1.])
    assert not safe and static < 0.
    assert diagnostic["reason"] == "SWEPT_COLLISION"


def test_swept_dynamic_segment_keeps_nonuniform_real_prediction_times():
    hull = tiny_hull()
    field = CollisionField([], hull)
    poses = np.zeros((2, 3))
    obstacle = MovingObstacle(-3., 0., 1., 0., .05)
    times = [2., 4.]
    assert moving_clearances(poses, hull, [obstacle], 0., times_s=times).min() > .8
    safe, _, dynamic, _ = certify_swept_poses(poses, field, [obstacle], times)
    assert not safe and dynamic < 0.


def test_refining_separate_intervals_does_not_create_an_unplanned_connecting_segment():
    field = CollisionField([(0., .12)], tiny_hull())
    poses = np.asarray([(-1., 0., 0.), (1., 0., 0.), (1., 1., 0.), (-1., 1., 0.)])
    safe, static, _, diagnostic = certify_swept_poses(poses, field, [], [0., 1., 2., 3.])
    assert safe and static == pytest.approx(.02)
    assert diagnostic["refinements"] >= 5
    assert diagnostic["static_lower_bound_m"] >= 0.


def test_expired_certification_budget_never_turns_unresolved_path_into_safe():
    field = CollisionField([(0., .12)], tiny_hull())
    poses = np.asarray([(-1., 0., 0.), (1., 0., 0.)])
    safe, _, _, diagnostic = certify_swept_poses(poses, field, [], [0., 1.], deadline=time.perf_counter() - 1.)
    assert not safe
    assert diagnostic["reason"] == "CERTIFICATION_BUDGET_REACHED"


def test_saved_coarse_safe_sequence_is_rejected_for_actual_intersample_intrusion():
    fixture, planner, field, moving = recorded(397904)
    state, disturbance = VesselState(**fixture["vessel"]), Disturbance(**fixture["disturbance"])
    controls = np.asarray(fixture["coarse_false_safe_controls_n"]) / 100.
    durations = fixture["coarse_false_safe_durations_s"]
    coarse = predict_sequence(planner.model, state, controls, durations, disturbance, 24, .2)
    assert field.clearances(coarse[:, :3]).min() > .005
    dense = predict_sequence(planner.model, state, controls, durations, disturbance, 240, .02)
    safe, static, _, _ = certify_swept_poses(dense[:, :3], field, moving, np.arange(241) * .02)
    assert not safe and static < -.015


def test_recorded_static_no_safe_feedback_has_a_certified_literal_force_escape():
    fixture, planner, field, moving = recorded()
    assert fixture["old_plan_status"] == "NO_SAFE_TRAJECTORY"
    state, disturbance = VesselState(**fixture["vessel"]), Disturbance(**fixture["disturbance"])
    plan = planner.choose(state, Guidance(**fixture["guidance"]), field, moving, disturbance)
    assert plan.status == "SAFE"
    assert planner.last_diagnostics["decision"] == "constant_force_escape"
    assert (plan.command.left_n, plan.command.right_n) == pytest.approx((400., 400.))
    assert plan.minimum_static_clearance_m > .3
    assert plan.minimum_dynamic_clearance_m > 0.
    certificate = planner.last_diagnostics["swept_certification"]
    assert certificate["static_lower_bound_m"] >= 0. and certificate["dynamic_lower_bound_m"] >= 0.


def test_navigation_force_escape_is_revalidated_at_actual_publication_state():
    fixture, planner, field, moving = recorded()
    state, disturbance = VesselState(**fixture["vessel"]), Disturbance(**fixture["disturbance"])
    previous = planner.model.actuator.command(**fixture["previous_force_n"])
    current = planner.model.step(state, previous, .05, disturbance)
    advanced = [replace(item, x=item.x + item.vx * .05, y=item.y + item.vy * .05) for item in moving]
    plan = planner.choose(state, Guidance(**fixture["guidance"]), field, moving, disturbance,
        execution_snapshot=lambda: (current, advanced, fixture["stamp_sim_s"] + .05))
    assert plan.status == "SAFE" and plan.trajectory[0] == current
    assert planner.last_diagnostics["selection_at_execution_state"]
    assert planner.last_diagnostics["prediction_stamp_sim_s"] == pytest.approx(fixture["stamp_sim_s"] + .05)
    assert field.clearances([item.pose for item in plan.trajectory]).min() > 0.
    assert moving_clearances([item.pose for item in plan.trajectory], field.hull, advanced, .2).min() > 0.


def test_new_dynamic_obstacle_can_revoke_previously_safe_constant_force():
    fixture, planner, field, moving = recorded()
    state, disturbance = VesselState(**fixture["vessel"]), Disturbance(**fixture["disturbance"])
    original = planner.model.actuator.command(400., 400.)
    route, _ = planner._force_trajectory(state, original, disturbance)
    obstacle = MovingObstacle(route[-1].x, route[-1].y, 0., 0., .3)
    assert moving_clearances([item.pose for item in route], field.hull, [obstacle], .2).min() < 0.
    plan = planner.choose(state, Guidance(**fixture["guidance"]), field, moving + [obstacle], disturbance)
    if plan.status == "SAFE":
        assert moving_clearances([item.pose for item in plan.trajectory], field.hull, moving + [obstacle], .2).min() >= 0.
        if plan.command == original:
            # 首段相同不等于四秒恒力；必须验证批准的实际换力确实避开新增目标。
            diagnostic = planner.last_diagnostics
            controls = np.asarray(diagnostic["selected_controls_n"])
            durations = diagnostic["selected_force_durations_s"]
            assert durations[0] < planner.horizon and np.any(controls[1:] != controls[0])
            dense = predict_sequence(planner.model, state, controls/100., durations, disturbance, 200, .02)
            assert field.clearances(dense[:, :3]).min() >= 0.
            assert moving_clearances(dense[:, :3], field.hull, moving + [obstacle], .02).min() >= 0.


def test_recorded_already_overlapping_hull_cannot_claim_escape_was_safe():
    fixture, planner, field, moving = recorded(411104)
    state = VesselState(**fixture["vessel"])
    assert field.clearances([state.pose])[0] < -.1
    plan = planner.choose(state, Guidance(**fixture["guidance"]), field, moving, Disturbance(**fixture["disturbance"]))
    assert plan.status != "SAFE" and plan.safe_candidate_count == 0
    assert plan.minimum_static_clearance_m < 0.


def test_normal_navigation_does_not_evaluate_emergency_literal_forces():
    _, planner, field, _ = recorded()
    empty = CollisionField([], field.hull)
    plan = planner.choose(VesselState(0., 0., 0., .4, 0., 0.), Guidance((20., 0.), .8, 0.), empty)
    assert plan.status == "SAFE"
    assert planner.last_diagnostics["constant_force_candidates_evaluated"] == 0
    assert planner.last_diagnostics["emergency_candidates_evaluated"] == 0


def test_no_safe_force_pool_does_not_sacrifice_immediate_dynamic_clearance_for_far_future():
    fixture, planner, field, original_moving = recorded(381504)
    state, disturbance = VesselState(**fixture["vessel"]), Disturbance(**fixture["disturbance"])
    applied = planner.model.actuator.command(**fixture["previous_force_n"])
    guidance = Guidance(**fixture["guidance"])
    actual_dynamic = []

    def advance(initial, command, begin, duration):
        count = math.ceil(duration / .02)
        for index in range(count):
            initial = planner.model.step(initial, command, duration / count, disturbance)
            stamp = begin + (index + 1) * duration / count
            moved = [replace(item, x=item.x + item.vx * stamp, y=item.y + item.vy * stamp) for item in original_moving]
            actual_dynamic.append(float(moving_clearances([initial.pose], field.hull, moved, .02).min()))
            assert field.clearances([initial.pose])[0] >= 0.
        return initial

    for step in range(25):
        elapsed = step * .12
        error = math.atan2(guidance.goal_xy[1] - state.y, guidance.goal_xy[0] - state.x) - state.yaw
        error = math.atan2(math.sin(error), math.cos(error))
        request = replace(guidance, desired_surge=.85 * max(0., math.cos(error)) ** 2,
                          desired_yaw_rate=max(-.5, min(.5, .8 * error)))
        current = state
        for _ in range(3):
            current = planner.model.step(current, applied, .05 / 3., disturbance)
        moved = [replace(item, x=item.x + item.vx * elapsed, y=item.y + item.vy * elapsed) for item in original_moving]
        advanced = [replace(item, x=item.x + item.vx * .05, y=item.y + item.vy * .05) for item in moved]
        plan = planner.choose(state, request, field, moved, disturbance,
            execution_snapshot=lambda: (current, advanced, elapsed + .05))
        state = advance(state, applied, elapsed, .05)
        state = advance(state, plan.command, elapsed + .05, .07)
        applied = plan.command
    assert min(actual_dynamic) > 0.
