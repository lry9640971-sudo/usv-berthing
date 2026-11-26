import math

import numpy as np
from scipy.spatial.transform import Rotation

from phase17_berthing_perception.berthing_core import (
    BasePose,
    base_pose_to_world,
    infer_bay_occupancy,
    make_berth_candidates,
    world_pose_to_base,
    leveling_rotation,
)


def test_world_base_round_trip():
    vehicle = BasePose(10.0, -4.0, 0.0, 0.7, 0.0, 0.0)
    local = world_pose_to_base(21.0, 8.0, -1.2, vehicle)
    world = base_pose_to_world(local["x"], local["y"], local["yaw"], vehicle)
    assert np.allclose(world[:2], (21.0, 8.0), atol=1e-9)
    assert abs(math.atan2(math.sin(world[2] + 1.2), math.cos(world[2] + 1.2))) < 1e-9


def test_three_bays_are_emitted_without_selection():
    occupancies = [{"status": "FREE", "confidence": 0.9, "cells": 0}] * 3
    rows = make_berth_candidates(
        (10.0, 2.0, 0.0),
        occupancies,
        valid=True,
        confidence=0.8,
        covariance=[0.0] * 9,
        reason="POINTCLOUD_GEOMETRY_VALIDATED",
    )
    assert [row["bay_id"] for row in rows] == [1, 2, 3]
    assert all(row["valid"] for row in rows)
    assert all("entrance_pose" in row and "final_pose" in row for row in rows)


def test_synthetic_object_marks_only_matching_bay_occupied():
    # Dock 与 base 重合，合成目标位于第二泊位内部且避开船体自遮罩。
    points = np.asarray([[1.0 + 0.25 * index, 9.0 + 0.05 * index, 0.8] for index in range(8)])
    rows = infer_bay_occupancy(points, (0.0, 0.0, 0.0), minimum_cells=5)
    assert rows[1]["status"] == "OCCUPIED"
    assert rows[0]["status"] == "UNKNOWN"
    assert rows[2]["status"] == "UNKNOWN"


def test_unrelated_far_point_cannot_prove_all_bays_free():
    rows = infer_bay_occupancy(np.asarray([[30., 30., 1.]]), (0., 0., 0.))
    assert all(row["status"] == "UNKNOWN" for row in rows)


def test_finite_rays_behind_one_bay_confirm_only_observed_bay():
    points = np.asarray([[5., y, 1.5] for y in np.linspace(-1., 7., 300)])
    origins = np.repeat([[-8., 3., 1.5]], len(points), axis=0)
    rows = infer_bay_occupancy(points, (0., 0., 0.), ray_origins_base=origins)
    assert rows[0]["status"] == "FREE"
    assert rows[0]["observed_fraction"] >= .65
    assert rows[2]["status"] == "UNKNOWN"


def test_leveling_undoes_body_tilt_instead_of_doubling_it():
    rotation = Rotation.from_euler("xyz", [.12, -.09, 0.]).as_matrix()
    leveled = np.asarray([[10., 4., .7]])
    body = leveled @ rotation
    assert np.allclose(body @ leveling_rotation(.12, -.09).T, leveled)
