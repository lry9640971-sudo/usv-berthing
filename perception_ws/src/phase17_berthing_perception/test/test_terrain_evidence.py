"""水面、平缓岸坡和船体倾斜的独立合成回归。"""

import numpy as np
from scipy.spatial.transform import Rotation
from phase17_berthing_perception.terrain_evidence import terrain_evidence


def scene():
    xw, yw = np.meshgrid(np.arange(-8., 8., .20), np.arange(3., 6., .20))
    xl, yl = np.meshgrid(np.arange(-8., 8., .20), np.arange(8., 15., .20))
    water = np.column_stack((xw.ravel(), yw.ravel(), np.full(xw.size, .20)))
    land = np.column_stack((xl.ravel(), yl.ravel(), .20 + .07 * (yl.ravel() - 6.)))
    return water, land


def test_flat_water_does_not_create_land_or_inferred_boundary():
    water, _ = scene()
    count = len(water)
    mask, land, edges, stats = terrain_evidence(water, water[:, 2], np.ones(count), np.ones(count, bool), np.ones(count, bool), .20)
    assert mask.all()
    assert not land.any() and not len(edges)


def test_shallow_rising_shore_is_kept_and_boundary_inferred_from_slope():
    water, land = scene()
    points = np.concatenate((water, land))
    count = len(points)
    mask, retained, edges, stats = terrain_evidence(points, points[:, 2], np.ones(count), np.ones(count, bool), np.ones(count, bool), .20)
    assert not retained[:len(water)].any()
    assert retained[len(water):].sum() > 1000
    assert len(edges) > 100
    assert np.allclose(edges[:, 1], 6. + .02 / .07, atol=.03)


def test_tilted_body_observation_uses_gravity_and_preserves_water_level():
    water, land = scene()
    points = np.concatenate((water, land))
    rotation = Rotation.from_euler("xyz", [.10, -.08, 0.]).as_matrix()
    body = (points - [0., 0., .15]) @ rotation
    elevation = body @ rotation[2, :] + .15
    count = len(body)
    mask, retained, edges, stats = terrain_evidence(body, elevation, np.ones(count), np.ones(count, bool), np.ones(count, bool), .20)
    assert not retained[:len(water)].any()
    assert abs(stats["water_elevation_m"] - .20) < 1e-6
    assert retained[len(water):].sum() > 1000


def test_sparse_or_vertical_returns_do_not_invent_water_reference():
    water, _ = scene()
    count = len(water)
    _, land, edges, stats = terrain_evidence(water, water[:, 2], np.zeros(count), np.ones(count, bool), np.ones(count, bool))
    assert not stats["observed_water_reference"]
    assert not land.any() and not len(edges)


def test_low_flat_land_is_not_reclassified_as_water_without_sea_returns():
    raised_platform, _ = scene()
    count = len(raised_platform)
    mask, retained, edges, stats = terrain_evidence(raised_platform, raised_platform[:, 2], np.ones(count), np.ones(count, bool), np.ones(count, bool), 0.)
    assert retained.all()
    assert not mask.any() and not stats["observed_water_reference"]
    assert not len(edges)
