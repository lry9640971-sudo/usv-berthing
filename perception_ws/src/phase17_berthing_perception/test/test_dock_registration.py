"""用已知几何与独立噪声检验局部配准，不能当作官方现场成绩。"""

from pathlib import Path
import numpy as np
import time

from phase17_berthing_perception.dock_localizer import load_template, rotate, register_with_coarse_prior


def test_noisy_prior_and_clutter_preserve_precise_measured_dock():
    path = Path(__file__).resolve().parents[1] / "config/full_dock_collision_points.csv"
    template, groups = load_template(path)
    rng = np.random.default_rng(20261003)
    points = rotate(template, -.5) + (12., 3.) + rng.normal(0., .025, size=template.shape)
    points = np.concatenate((points[::3], rng.uniform((-20., -20.), (30., 30.), (1000, 2))))
    rows, status = register_with_coarse_prior(points, template, groups, {"x": 13., "y": 4., "yaw": -.45}, beam_width=3)
    assert status == "UNIQUE"
    assert np.linalg.norm(np.asarray([rows[0]["x"], rows[0]["y"]]) - (12., 3.)) < .15
    assert abs(rows[0]["yaw"] + .5) < .01


def test_expired_registration_budget_does_not_refine_all_hypotheses():
    path = Path(__file__).resolve().parents[1] / "config/full_dock_collision_points.csv"
    template, groups = load_template(path)
    rows, status = register_with_coarse_prior(np.empty((0, 2)), template, groups,
                                             {"x": 10., "y": 0., "yaw": 0.}, deadline=time.perf_counter() - 1.)
    assert not rows
    assert status == "NO_ESTIMATE"
