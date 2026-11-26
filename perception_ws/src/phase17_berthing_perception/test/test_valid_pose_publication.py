"""直接执行生产帧处理方法，确认无效结果不会进入正式位姿话题。"""

from collections import defaultdict
from threading import RLock
from types import SimpleNamespace as S
import numpy as np

from phase17_berthing_perception.berthing_perception_node import BerthingPerceptionNode
from phase16z_s3_runtime.s3_timestamped_frames import PlanarPose


def test_empty_geometry_publishes_degraded_contract_without_dock_pose(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("无效码头不得执行占用射线计算")
    monkeypatch.setattr("phase17_berthing_perception.berthing_perception_node.infer_bay_occupancy", forbidden)
    node = S()
    node.get_clock = lambda: S(now=lambda: S(nanoseconds=10_000_000_000))
    node.lock, node.counts = RLock(), defaultdict(int)
    node.output_stale_s, node.maximum_processing_s = .5, .4
    node.extrinsics = {}
    node.self_geometry = S(contains=lambda points: np.zeros(len(points), dtype=bool))
    node.template = np.asarray([[0., 0.], [5., 0.], [5., 18.]])
    node.template_groups = {"BACK_WALL": node.template}
    node.maximum_dock_range_m = 35.
    node.dock_yaw_override = 0.
    node.use_topic_dock_orientation = True
    node.geometry_score_threshold, node.minimum_geometry_support, node.minimum_active_groups = .18, .65, 2
    node.occupancy_minimum_cells, node.entrance_offset_m = 5, -8.
    node.base_frame, node.world_frame = "wamv/base_link", "team_enu"
    outputs = {"local": [], "world": [], "pose": []}
    node.local_publisher = S(publish=outputs["local"].append)
    node.world_publisher = S(publish=outputs["world"].append)
    node.pose_publisher = S(publish=outputs["pose"].append)
    quaternion = S(x=0., y=0., z=0., w=1.)
    snapshot = {"anchor": 10_000_000_000, "clouds": [],
                "poses": {10_000_000_000: PlanarPose(10_000_000_000, 0., 0., 0., 0., (0., 0., 0., 1.))},
                "dock": S(pose=S(orientation=quaternion, position=S(x=1000., y=0., z=0.))),
                "input_mode": "FULL_SYNC", "sync_span_ms": 0.}
    BerthingPerceptionNode._process_snapshot(node, snapshot)
    assert outputs["world"] and '"status":"DEGRADED"' in outputs["world"][0].data
    assert not outputs["pose"]
