"""完整真实部件边界索引与原GEOS距离独立对照，包含内部及细窄间隙。"""

from pathlib import Path

import numpy as np
import pytest
import shapely

from competition_user.closed_loop.geometry import Hull

ROOT = Path(__file__).resolve().parents[1]


def test_all_original_union_edges_keep_exact_signed_distance():
    hull = Hull.load(ROOT/'perception_ws/src/phase17_berthing_perception/config/hull_collision_footprint_base_link.yaml')
    random = np.random.default_rng(61004)
    points = random.uniform((-4., -2.), (4., 2.), (30000, 2))
    boundary = shapely.get_coordinates(hull.boundary)
    points = np.concatenate((points, boundary, boundary+[1e-9, -1e-9]))
    reference = shapely.distance(hull.boundary, shapely.points(points))
    reference[shapely.contains_xy(hull.geometry, points[:, 0], points[:, 1])] *= -1.
    assert hull.signed_distance(points) == pytest.approx(reference, abs=2e-12)


def test_boundary_index_keeps_holes_and_disjoint_components():
    # 四条实物围成的孔洞和旁边独立实物不能被空间索引或凸包填实。
    parts = [np.asarray(p) for p in (
        ((-2., -2.), (2., -2.), (2., -1.), (-2., -1.)),
        ((-2., 1.), (2., 1.), (2., 2.), (-2., 2.)),
        ((-2., -1.), (-1., -1.), (-1., 1.), (-2., 1.)),
        ((1., -1.), (2., -1.), (2., 1.), (1., 1.)),
        ((4., 0.), (5., 0.), (5., 1.), (4., 1.)),
    )]
    hull = Hull(np.concatenate(parts), parts)
    points = np.asarray(((0., 0.), (1.5, 0.), (4.5, .5), (3., 0.), (1., 0.)))
    assert hull.signed_distance(points) == pytest.approx((1., -.5, -.5, 1., 0.), abs=1e-12)
