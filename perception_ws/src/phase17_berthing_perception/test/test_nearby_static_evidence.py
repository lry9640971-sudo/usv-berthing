"""船体边界外的近场立面不能被扩大后的本船矩形删除。"""

import numpy as np

from phase17_berthing_perception.environment_core import extract_cloud_evidence
from phase17_berthing_perception.grid_core import GridConfig


def test_external_vertical_surface_is_not_ownship():
    yy, zz = np.meshgrid(np.linspace(1.3, 1.5, 11), np.linspace(.9, 1.7, 12))
    cloud = np.column_stack((np.full(yy.size, 2.0), yy.ravel(), zz.ravel()))
    footprint = (np.asarray(((-2.4, -1.23), (1.6, -1.23), (1.6, 1.23), (-2.4, 1.23))),)
    occupied, _, _, stats = extract_cloud_evidence(cloud, width=11, height=12,
        transform=np.eye(4), footprint_polygons=footprint, cfg=GridConfig(),
        navigation_z_m=0., gravity_rotation=np.eye(3))
    assert len(occupied)>30
    assert stats["self_points"]==0
