"""验证转向及横纵摇时，真实占用和码头融合不会随本船画出假障碍。"""

from collections import defaultdict
import json
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

from phase16z_s3_runtime.s3_timestamped_frames import PlanarPose
from phase17_berthing_perception.grid_core import GridConfig
from phase17_berthing_perception.local_environment_node import LocalEnvironmentNode, DOCK_TOPIC
from phase17_berthing_perception.navigation_history import planar_output_transform
from competition_user.closed_loop.navigation import ObservationMap


def pose(stamp, xyz, rpy):
    return PlanarPose(round(stamp*1e9), *xyz[:2], rpy[2], xyz[2],
                      tuple(Rotation.from_euler('xyz', rpy).as_quat()))


def test_projected_cloud_stays_at_same_world_xy_across_full_3d_motion():
    world = np.asarray(((12., 7., .5), (11., 9., 1.7), (15., 6., .3)))
    source = pose(1., (10., 8., .2), (.08, .101, .35))
    anchor = pose(1.3, (10.4, 8.2, .25), (-.1, -.08, .53))
    source_rotation = Rotation.from_quat(source.quaternion_xyzw).as_matrix()
    sensor_base = (world-(source.x_enu_m, source.y_enu_m, source.z_m))@source_rotation
    for target in (source, anchor):
        matrix = planar_output_transform(source, target)
        projected = sensor_base@matrix[:3, :3].T+matrix[:3, 3]
        yaw = Rotation.from_euler('z', target.yaw_rad).as_matrix()[:2, :2]
        restored = projected[:, :2]@yaw.T+[target.x_enu_m, target.y_enu_m]
        assert restored == pytest.approx(world[:, :2], abs=1e-12)


def environment_stub():
    published = []
    sink = SimpleNamespace(publish=lambda message: None)
    cfg = GridConfig(width=80, height=80, origin_xy=(-8., -8.))
    node = SimpleNamespace(cfg=cfg, lock=threading.RLock(), latest_dock=None,
        dock_fusion_stale_ns=500_000_000, dock_template=np.asarray(((0., 0.),)),
        self_mask=np.zeros((cfg.height, cfg.width), dtype=bool), inflation_radius_m=0.,
        minimum_occupied_cell_hits=2, grid_publisher=SimpleNamespace(publish=published.append),
        centers_publisher=sink, obstacles_publisher=sink, base_frame='wamv/base_link_planar',
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_400_000_000)),
        counts=defaultdict(int))
    return node, published


def test_live_environment_fuses_world_dock_at_grid_stamp_not_old_ship_frame():
    node, published = environment_stub()
    assert DOCK_TOPIC == '/team/perception/berth_candidates_world'
    payload = {'frame_id': 'team_enu', 'status': 'READY', 'stamp_ns': 1_000_000_000,
               'dock_pose_xyyaw_m_rad': [12., 7., 0.]}
    LocalEnvironmentNode._dock_callback(node, String(data=json.dumps(payload)))
    observed = ObservationMap()
    for anchor in (pose(1.3, (10., 8., .2), (.08, .1, .5)),
                   pose(1.4, (10.4, 8.2, .3), (-.1, .07, .7))):
        LocalEnvironmentNode._process_snapshot(node, {'poses': {anchor.stamp_ns: anchor},
            'anchor': anchor.stamp_ns, 'clouds': [], 'span_ms': 0., 'input_mode': 'PARTIAL_FRESH'})
        message = published[-1]
        assert message.header.frame_id == 'wamv/base_link_planar'
        grid = np.asarray(message.data).reshape(message.info.height, message.info.width)
        assert (grid == 100).sum() == 1
        observed.update(grid, (*node.cfg.origin_xy, 0.),
                        (anchor.x_enu_m, anchor.y_enu_m, anchor.yaw_rad), node.cfg.resolution)
    points = observed.nearby((12., 7.))
    # 旋转后的格中心量化误差不大于单格半对角，不能把旧船体转动累积成假墙。
    assert np.max(np.linalg.norm(points-(12., 7.), axis=1)) <= math.sqrt(2)*node.cfg.resolution/2+1e-12


def test_dock_fusion_rejects_local_pose_on_world_protocol():
    node, _ = environment_stub()
    payload = {'frame_id': 'wamv/base_link', 'status': 'READY', 'stamp_ns': 1_000_000_000,
               'dock_pose_xyyaw_m_rad': [2., 0., 0.]}
    LocalEnvironmentNode._dock_callback(node, String(data=json.dumps(payload)))
    assert not node.latest_dock['valid']


def test_dock_padding_is_not_accumulated_as_physical_geometry():
    node, published = environment_stub()
    node.inflation_radius_m = .4
    anchor = pose(1.3, (10., 8., .2), (.08, .1, .5))
    node.latest_dock = {'stamp_ns': 1_000_000_000, 'pose': [12., 7., 0.], 'valid': True}
    LocalEnvironmentNode._process_snapshot(node, {'poses': {anchor.stamp_ns: anchor},
        'anchor': anchor.stamp_ns, 'clouds': [], 'span_ms': 0., 'input_mode': 'PARTIAL_FRESH'})
    grid = np.asarray(published[-1].data).reshape(node.cfg.height, node.cfg.width)
    assert (grid == 100).sum() == 1
    assert (grid == 50).sum() > 1
    observed = ObservationMap()
    observed.update(grid, (*node.cfg.origin_xy, 0.),
                    (anchor.x_enu_m, anchor.y_enu_m, anchor.yaw_rad), node.cfg.resolution)
    assert len(observed.occupied) == 1
