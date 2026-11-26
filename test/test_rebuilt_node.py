"""用真实 ROS 消息验证新入口适配、实际推力及完成事件，不借用旧状态机。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import rclpy
from nav_msgs.msg import Odometry, OccupancyGrid
from phase16z_perception_msgs.msg import DynamicTrackArray
from std_msgs.msg import String

import competition_user.closed_loop.node as module
from phase16z_s3_runtime.s3_runtime_contracts import BASE_FRAME, stamp_to_msg


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controller(tmp_path, monkeypatch):
    directories = {"competition_user": ROOT / "control_ws/src/competition_user",
                   "phase17_berthing_perception": ROOT / "perception_ws/src/phase17_berthing_perception"}
    monkeypatch.setattr(module, "get_package_share_directory", lambda package: str(directories[package]))
    rclpy.init(args=["--ros-args", "-p", "runtime_output_dir:=" + str(tmp_path)])
    node = module.ClosedLoopNode()
    try:
        node.test_now = 10.
        node.now_sim = lambda: node.test_now
        node.published = {key: [] for key in ("left", "right", "completion")}
        node.left = SimpleNamespace(publish=node.published["left"].append)
        node.right = SimpleNamespace(publish=node.published["right"].append)
        node.completion_publisher = SimpleNamespace(publish=node.published["completion"].append)
        yield node
    finally:
        node.execution_log.close()
        node.destroy_node()
        rclpy.shutdown()


def send_inputs(node, x=0., y=0., surge=.4, sway=.05):
    odom = Odometry()
    odom.header.frame_id, odom.child_frame_id = "team_enu", BASE_FRAME
    odom.header.stamp = stamp_to_msg(int(node.test_now * 1e9))
    odom.pose.pose.orientation.w = 1.
    odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
    odom.twist.twist.linear.x, odom.twist.twist.linear.y = surge, sway
    node.odometry(odom)
    grid = OccupancyGrid()
    grid.header.frame_id, grid.header.stamp = BASE_FRAME, odom.header.stamp
    grid.info.width = grid.info.height = 4
    grid.info.resolution = .2
    grid.info.origin.orientation.w = 1.
    grid.data = [0] * 16
    node.occupancy(grid)
    tracks = DynamicTrackArray()
    tracks.header.frame_id, tracks.header.stamp = "team_enu", odom.header.stamp
    node.tracks(tracks)
    return grid


def send_bay(node):
    row = {"bay_id": 1, "valid": True, "occupancy_status": "FREE",
           "entry_pose": {"x": -4., "y": 3., "yaw": 0.},
           "final_pose": {"x": 1.5, "y": 3., "yaw": 0.}}
    payload = {"frame_id": "team_enu", "stamp_ns": int(node.test_now * 1e9), "status": "READY",
               "dock_pose_xyyaw_m_rad": [0., 0., 0.], "candidates": [row]}
    node.bays(String(data=json.dumps(payload)))


def test_ros_inputs_publish_the_same_force_as_saved_prediction(controller):
    node = controller
    send_inputs(node)
    node.mission.dock_prior = (30., 0., 0.)
    node.tick()
    node.execution_log.flush()
    row = json.loads((node.output / "execution_log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["input_status"] == "READY"
    assert row["plan_status"] == "SAFE"
    assert node.published["left"][-1].data == pytest.approx(row["published_force_n"]["left_n"])
    assert node.published["right"][-1].data == pytest.approx(row["published_force_n"]["right_n"])
    assert row["prediction_trajectory"][0]["sway"] == pytest.approx(.05)
    assert node.applied == node.last_plan.command


def test_bad_grid_does_not_refresh_input_or_crash(controller):
    grid = send_inputs(controller)
    grid.header.stamp = stamp_to_msg(10_100_000_000)
    grid.data = [0]
    controller.test_now = 10.1
    controller.occupancy(grid)
    assert controller.receipts["grid"][0] == 10.
    assert controller.rejected["GRID_LAYOUT"] == 1


def test_horizontal_grid_frame_is_accepted_without_changing_odometry_frame(controller):
    grid = send_inputs(controller)
    controller.test_now = 10.1
    grid.header.frame_id = "wamv/base_link_planar"
    grid.header.stamp = stamp_to_msg(10_100_000_000)
    controller.occupancy(grid)
    assert controller.receipts["grid"][0] == pytest.approx(10.1)
    assert "wamv/base_link_planar" not in module.BASE_FRAMES


def test_completion_event_contains_measurement_instant_and_real_bay(controller):
    node = controller
    send_inputs(node, x=-4., y=3., surge=0., sway=0.)
    send_bay(node)
    node.tick()
    for stamp in (10.2, 11.3):
        node.test_now = stamp
        send_inputs(node, x=1.5, y=3., surge=0., sway=0.)
        send_bay(node)
        node.tick()
    report = json.loads((node.output / "completion.json").read_text(encoding="utf-8"))
    event = json.loads(node.published["completion"][0].data)
    assert report["task_completed"] and report["bay_id"] == 1
    assert event["stamp_sim_s"] == pytest.approx(10.2)
    assert len(node.published["completion"]) == 1
    assert event["position_xy_m"] == pytest.approx([1.5, 3.])
    assert report["max_control_cycle_s"] > 0.


def test_cycle_timing_includes_status_output(controller, monkeypatch):
    node = controller
    clock = [0.]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    node.write_status = lambda: clock.__setitem__(0, .55)
    node.tick()
    assert node.max_cycle_s == pytest.approx(.55)
