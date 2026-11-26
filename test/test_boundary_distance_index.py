"""精确最近段索引与原完整部件并集距离的等价性及真实预算反例。"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import shapely

from competition_user.closed_loop.dynamics import Disturbance, VesselDynamics, VesselState
from competition_user.closed_loop.geometry import CollisionField, Hull
from competition_user.closed_loop.planner import Guidance, MovingObstacle, ThrustPlanner

ROOT = Path(__file__).resolve().parents[1]


def actual_hull():
    return Hull.load(ROOT/"perception_ws/src/phase17_berthing_perception/config/hull_collision_footprint_base_link.yaml")


def original_distance(hull, points):
    points = np.asarray(points).reshape(-1, 2)
    if not len(points):
        return np.empty(0)
    indices, nearest = hull.boundary_index.query_nearest(shapely.points(points),
        return_distance=True, all_matches=False)
    distances = np.full(len(points), np.nan)
    distances[indices[0]] = nearest
    return np.where(shapely.contains_xy(hull.geometry, points[:, 0], points[:, 1]), -distances, distances)


def test_index_matches_original_full_boundary_for_interior_gaps_and_far_field():
    hull = actual_hull()
    random = np.random.default_rng(741)
    points = random.uniform(-12., 12., (30000, 2))
    boundaries = np.concatenate([np.asarray(ring.coords) for ring in shapely.get_parts(hull.boundary)])
    points = np.concatenate((points, boundaries, boundaries+[1e-8, -1e-8], [[np.nan, 0.]]))
    actual, expected = hull.signed_distance(points), original_distance(hull, points)
    assert np.allclose(actual, expected, atol=1e-12, rtol=0., equal_nan=True)
    assert np.any(expected > 0.) and np.any(expected < 0.)
    assert len(hull.signed_distance([])) == 0


def test_cell_corner_rounding_keeps_all_possible_nearest_segments():
    hull = actual_hull()
    cache = hull.distance_index
    y, x = np.indices((cache.width+1, cache.width+1))
    corners = cache.origin+cache.cell_size*np.column_stack((x.ravel(), y.ravel()))
    points = np.concatenate((corners, np.nextafter(corners, np.inf), np.nextafter(corners, -np.inf)))
    assert hull.signed_distance(points) == pytest.approx(original_distance(hull, points), abs=1e-12)


def recorded_case():
    fixture = json.loads((ROOT/"test/fixtures/terminal_budget_465704_224214.json").read_text(encoding="utf-8"))
    data = np.load(ROOT/"test/fixtures/terminal_budget_465704_224214.npz", allow_pickle=False)
    model = VesselDynamics.load(ROOT/"control_ws/src/competition_user/config/vessel_dynamics.yaml")
    hull = actual_hull()
    field = CollisionField(data["points"], hull, float(data["point_radius_m"]))
    return fixture, model, field


def test_dense_actual_terminal_map_keeps_original_exact_clearances():
    fixture, model, field = recorded_case()
    poses = np.asarray([[item["x"], item["y"], item["yaw"]] for item in fixture["recorded_trajectory"]])
    random = np.random.default_rng(142)
    poses = np.concatenate([poses+random.uniform([-.15, -.15, -.06], [.15, .15, .06], poses.shape)
                            for _ in range(8)])
    cached = field.clearances(poses)
    field.hull.signed_distance = lambda points: original_distance(field.hull, points)
    assert cached == pytest.approx(field.clearances(poses), abs=1e-12)


def test_actual_over_budget_snapshot_preserves_no_safe_and_real_reverse_force():
    fixture, model, field = recorded_case()
    planner = ThrustPlanner(model, field.hull)
    prior = fixture["previous_decision"]
    planner.terminal_optimizer.previous = np.asarray(prior["warm_start_controls_n"])/100.
    planner.terminal_optimizer.previous_sequence = (np.asarray(prior["selected_controls_n"])/100.,
        np.asarray(prior["selected_force_durations_s"]), prior["prediction_stamp_sim_s"])
    stamp = fixture["prediction_stamp_sim_s"]
    age = stamp-fixture["stamp_sim_s"]
    moving = [MovingObstacle(**{k: v for k, v in item.items() if k != "track_id"}) for item in fixture["moving_obstacles"]]
    advanced = [replace(item, x=item.x+item.vx*age, y=item.y+item.vy*age) for item in moving]
    current = VesselState(**fixture["prediction_initial_vessel"])
    plan = planner.choose(VesselState(**fixture["vessel"]), Guidance(**fixture["guidance"]), field, moving,
        Disturbance(**fixture["disturbance"]), execution_snapshot=lambda: (current, advanced, stamp))
    assert plan.status == "NO_SAFE_TRAJECTORY" and plan.safe_candidate_count == 0
    assert plan.command.__dict__ == fixture["recorded_force_n"]
    assert plan.minimum_static_clearance_m == pytest.approx(fixture["recorded_static_m"], abs=1e-12)
