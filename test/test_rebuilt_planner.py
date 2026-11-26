"""真实尺寸、侧向漂移、中途相遇和最终指令一致性的闭环反例。"""

import math
import json
from pathlib import Path

import numpy as np
import pytest

from competition_user.closed_loop.dynamics import VesselDynamics, VesselState, Disturbance
from competition_user.closed_loop.geometry import Hull, CollisionField, moving_clearances
from competition_user.closed_loop.planner import ThrustPlanner, Guidance, MovingObstacle
from competition_user.closed_loop.navigation import RoutePlanner
from phase17_berthing_perception.berthing_core import make_berth_candidates


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "perception_ws/src/phase17_berthing_perception/config"


def model_and_hull():
    return (VesselDynamics.load(ROOT / "control_ws/src/competition_user/config/vessel_dynamics.yaml"),
            Hull.load(CONFIG / "hull_collision_footprint_base_link.yaml"))


def test_native_contour_distances_preserve_actual_polygon_to_one_micrometre():
    _, hull = model_and_hull()
    hull = Hull(hull.vertices)
    points = np.random.default_rng(13).uniform([-4., -3.], [4., 3.], (2000, 2))
    planes = points @ hull.normals.T + hull.offsets
    difference = points[:, None, :] - hull.vertices
    fraction = np.clip(np.sum(difference*hull.edges, axis=2)/hull.edge_length_sq, 0., 1.)
    closest = hull.vertices + fraction[:, :, None]*hull.edges
    distance = np.sqrt(np.sum((points[:, None, :]-closest)**2, axis=2).min(axis=1))
    reference = np.where(planes.max(axis=1) <= 0., -distance, distance)
    assert hull.signed_distance(points) == pytest.approx(reference, abs=1e-6)


def test_full_physical_components_keep_engine_gap_without_erasing_engine():
    _, hull = model_and_hull()
    convex = Hull(hull.vertices)
    gap = (-2.5, -.9)
    assert convex.signed_distance([gap])[0] < -.3
    assert hull.signed_distance([gap])[0] > .05
    assert hull.signed_distance([(-2.7, -1.027135)])[0] < 0.


def test_filtered_distance_matches_all_physical_parts_in_rotated_near_field():
    _, hull = model_and_hull()
    random = np.random.default_rng(41)
    points = random.uniform([-6., -5.], [6., 5.], (250, 2))
    poses = random.uniform([-1., -1., -3.], [1., 1., 3.], (30, 3))
    field = CollisionField(points, hull, .14)
    reference = []
    for x, y, yaw in poses:
        c, s = math.cos(yaw), math.sin(yaw)
        local = (points-(x, y)) @ np.asarray(((c, -s), (s, c)))
        reference.append(hull.signed_distance(local).min()-.14)
    assert field.clearances(poses) == pytest.approx(reference, abs=1e-9)


def test_fast_halfspace_bound_never_discards_a_closer_physical_obstacle():
    _, hull = model_and_hull()
    points = np.random.default_rng(92).uniform([-6., -4.], [6., 4.], (12000, 2))
    bound = hull.convex_lower_distance(points)
    assert np.all(bound <= hull.signed_distance(points)+1e-10)
    field = CollisionField(points, hull, .14)
    poses = np.asarray([(0., 0., yaw) for yaw in np.linspace(-3., 3., 9)])
    reference = []
    for x, y, yaw in poses:
        local = (points-(x, y))@np.asarray(((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw))))
        reference.append(hull.signed_distance(local).min()-.14)
    assert field.clearances(poses) == pytest.approx(reference, abs=1e-9)


def test_blocked_coarse_start_does_not_create_fake_stationary_route():
    _, hull = model_and_hull()
    router = RoutePlanner(hull)
    assert router.plan((0., 0.), (10., 0.), [(0., 1.5)], 1.) == []
    assert router.status == "COARSE_START_BLOCKED"
    assert router.lookahead((0., 0.)) is None


def test_real_boat_fits_final_bay_along_entrance_but_not_old_ninety_degree_yaw():
    _, hull = model_and_hull()
    dock = np.loadtxt(CONFIG / "full_dock_collision_points.csv", delimiter=",", skiprows=1, usecols=(0, 1))
    field = CollisionField(dock, hull)
    occupancy = [{"status": "FREE", "confidence": 1., "cells": 100}] * 3
    bays = make_berth_candidates((0., 0., 0.), occupancy, valid=True, confidence=1., covariance=[0.] * 9, reason="VALID")
    for bay in bays:
        pose = tuple(bay["final_pose"][key] for key in ("x", "y", "yaw"))
        assert field.clearances([pose])[0] > .5
        assert field.clearances([(pose[0], pose[1], math.pi / 2)])[0] < -.1


def test_returned_force_is_first_actual_prediction_command():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    state = VesselState(0., 0., 0., .4, .05, .02)
    disturbance = Disturbance(10., 5., -3.)
    field = CollisionField([], hull)
    plan = planner.choose(state, Guidance((10., 0.), .8, 0.), field, disturbance=disturbance)
    assert plan.status == "SAFE"
    assert plan.command == model.speed_command(state, *plan.setpoint, disturbance=disturbance)
    assert model.actuator.published_values(plan.command) == (plan.command.left_n, plan.command.right_n)
    intermediate = model.step(state, plan.command, .1, disturbance)
    predicted = model.step(intermediate, plan.command, .1, disturbance)
    assert plan.trajectory[1] == predicted


def test_heading_feedback_prediction_stops_turning_at_requested_bearing():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    trajectory, _ = planner._trajectory(VesselState(0., 0., 0., 0., 0., 0.), (0., .3), Disturbance())
    assert .30 < trajectory[-1].yaw < .40
    assert abs(trajectory[-1].yaw_rate) < .04


def test_entry_alignment_can_move_base_with_com_rotation_inside_actual_tolerance():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    state = VesselState(-8., 3., -2.7, .02, .01, 0.)
    guidance = Guidance((-8., 3.), 0., .5, goal_yaw=0., use_route=False, position_tolerance_m=1.)
    plan = planner.choose(state, guidance, CollisionField([], hull), disturbance=Disturbance(10., 5., 2.))
    assert plan.status == "SAFE" and plan.setpoint[1] > .3
    assert abs(plan.trajectory[-1].yaw) < abs(state.yaw)-.4


def test_terminal_prediction_brakes_towards_goal_instead_of_holding_sample_speed():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    state = VesselState(0., 0., 0., .15, 0., 0.)
    guidance = Guidance((.5, 0.), .18, 0., 0., True, feedback="terminal")
    trajectory, commands = planner._trajectory(state, (.24, 0.), Disturbance(), guidance)
    assert commands[0] == model.speed_command(state, .24, 0.)
    assert .35 < trajectory[-1].x < .7
    assert trajectory[-1].surge < .10


def test_final_corridor_centers_against_observed_drift_before_near_field():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    state = VesselState(0., .2, 0., .45, .05, 0.)
    guidance = Guidance((10., 0.), .45, 0., 0., False, feedback="terminal")
    plan = planner.choose(state, guidance, CollisionField([], hull), disturbance=Disturbance(10., 5., 2.))
    assert plan.status == "SAFE" and plan.setpoint[1] < 0.
    assert abs(plan.trajectory[-1].y) < .2


def test_unavoidable_drift_in_current_candidates_is_reported_not_claimed_safe():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    state = VesselState(0., 0., 0., 0., .35, 0.)
    field = CollisionField([(x, 1.9) for x in np.arange(-6., 6., .2)], hull, .14)
    coast = model.predict(state, model.actuator.command(0., 0.), 4.)
    assert field.clearances([item.pose for item in coast]).min() < 0
    plan = planner.choose(state, Guidance((10., -3.), .6, -.3), field)
    assert plan.safe_candidate_count == 0
    assert plan.status == "NO_SAFE_TRAJECTORY"
    assert plan.minimum_static_clearance_m < 0


def test_local_pillar_drift_has_an_actual_departure_command():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    state = VesselState(0., 0., 0., 0., .35, 0.)
    field = CollisionField([(0., 1.9)], hull, .14)
    coast = model.predict(state, model.actuator.command(0., 0.), 4.)
    assert field.clearances([item.pose for item in coast]).min() < 0
    plan = planner.choose(state, Guidance((10., -3.), .6, -.3), field)
    assert plan.safe_candidate_count > 0
    assert plan.minimum_static_clearance_m >= 0
    assert plan.setpoint != (0., 0.)


def test_dynamic_mid_horizon_crossing_cannot_be_hidden_by_safe_endpoints():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    state = VesselState(0., 0., 0., .8, 0., 0.)
    field = CollisionField([], hull)
    obstacle = MovingObstacle(5., -10., 0., 5.)
    trajectory, _ = planner._trajectory(state, (.8, 0.), Disturbance())
    clearance = planner._dynamic_clearance(trajectory, [obstacle])
    assert clearance[0] > 0 and clearance[-1] > 0 and clearance.min() < 0
    plan = planner.choose(state, Guidance((15., 0.), .8, 0.), field, [obstacle])
    assert plan.safe_candidate_count > 0
    assert plan.minimum_dynamic_clearance_m >= 0


def test_dynamic_clearance_preserves_rotated_full_hull_and_original_uncertainty():
    _, hull = model_and_hull()
    random = np.random.default_rng(45)
    poses = random.uniform([-3., -3., -3.], [3., 3., 3.], (4, 21, 3))
    obstacles = [MovingObstacle(5., 1., -.4, .2, .8, .15), MovingObstacle(-2., 3., .5, -.4, 1.5, .3)]
    actual = moving_clearances(poses, hull, obstacles, .2)
    for batch in range(len(poses)):
        reference = []
        for step, (x, y, yaw) in enumerate(poses[batch]):
            values = []
            for item in obstacles:
                delta = np.asarray((item.x+.2*step*item.vx-x, item.y+.2*step*item.vy-y))
                local = delta@np.asarray(((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw))))
                values.append(hull.signed_distance([local])[0]-item.radius_m-.2*step*item.velocity_std_mps)
            reference.append(min(values))
        assert actual[batch] == pytest.approx(reference, abs=1e-9)


def test_recorded_boat37_state_can_escape_with_permitted_forces_before_turning_collision():
    model, hull = model_and_hull()
    fixture = json.loads((ROOT/"test/fixtures/contact_boat37_3878.json").read_text(encoding="utf-8"))
    state = VesselState(**fixture["vessel"])
    guidance = Guidance(**fixture["guidance"])
    disturbance = Disturbance(**fixture["disturbance"])
    obstacles = [MovingObstacle(*item["xy"], *item["velocity_xy"], velocity_std_mps=.4)
                 for item in fixture["public_tracks"]["tracks"]]
    planner = ThrustPlanner(model, hull)
    plan = planner.choose(state, guidance, CollisionField([], hull), obstacles, disturbance)
    assert fixture["plan_status"] == "NO_SAFE_TRAJECTORY"
    assert plan.status == "SAFE" and plan.minimum_dynamic_clearance_m > 0
    assert plan.setpoint[0] < -.6
    assert planner.last_diagnostics["decision"] == "actuation_range_escape"
    assert max(abs(plan.command.left_n), abs(plan.command.right_n)) <= 400.
    assert plan.command == model.speed_command(state, *plan.setpoint, disturbance=disturbance)
    # 保存反例不含旧地图，新增真实后方障碍仍必须让这条脱离动作失效。
    blocked = CollisionField([(-553., 210.), (-554., 211.), (-550., 209.), (-550., 205.)], hull, .14)
    denied = planner.choose(state, guidance, blocked, obstacles, disturbance)
    assert denied.status != "SAFE" or blocked.clearances([item.pose for item in denied.trajectory]).min() >= 0


def test_emergency_actuation_does_not_raise_normal_navigation_speed():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    plan = planner.choose(VesselState(0., 0., 0., .4, 0., 0.), Guidance((20., 0.), .8, 0.), CollisionField([], hull))
    assert plan.status == "SAFE" and plan.setpoint[0] <= .8
    assert planner.last_diagnostics["emergency_candidates_evaluated"] == 0


def test_tight_queries_preserve_original_sparse_far_field_distance_convention():
    _, hull = model_and_hull()
    points = np.asarray(((20., 8.), (19., -12.)))
    field = CollisionField(points, hull, .14)
    poses = np.asarray(((0., 0., 0.), (1., -1., 2.4), (-3., 4., -2.9)))
    expected = np.linalg.norm(points[None, :, :]-poses[:, None, :2], axis=2).min(axis=1)-hull.radius-.14
    assert field.clearances(poses) == pytest.approx(expected, abs=1e-9)
