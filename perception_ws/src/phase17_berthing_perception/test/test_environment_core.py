import numpy as np

from phase17_berthing_perception.environment_core import inflate_occupied, points_in_polygons
from phase17_berthing_perception.grid_core import GridConfig


def test_polygon_mask_uses_component_shape():
    polygon = np.asarray(((-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)))
    points = np.asarray(((0.0, 0.0), (1.2, 0.0), (0.0, 0.8)))
    assert points_in_polygons(points, (polygon,)).tolist() == [True, False, False]


def test_padding_is_circular_and_does_not_become_a_second_physical_obstacle():
    cfg = GridConfig(resolution=0.2, width=21, height=21, origin_xy=(-2.1, -2.1))
    grid = np.full((21, 21), -1, dtype=np.int8)
    grid[10, 10] = 100
    inflated = inflate_occupied(grid, cfg, 0.4)
    assert inflated[10, 10] == 100
    assert inflated[10, 12] == 50
    assert inflated[12, 12] == -1
    assert (inflated == 100).sum() == 1


def test_padding_preserves_measured_free_cells_for_map_clearing():
    cfg = GridConfig(resolution=0.2, width=21, height=21, origin_xy=(-2.1, -2.1))
    grid = np.full((21, 21), -1, dtype=np.int8)
    grid[10, 10] = 100
    grid[10, 11] = 0
    inflated = inflate_occupied(grid, cfg, 0.4)
    assert inflated[10, 11] == 0
    assert inflated[10, 9] == 50
