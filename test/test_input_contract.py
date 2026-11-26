"""跨模块反例：命令因果关系、地图清除与完整观测时刻几何。"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from competition_user.closed_loop.actuator import ThrustCommand
from competition_user.closed_loop.dynamics import DisturbanceEstimator, VesselDynamics, VesselState
from competition_user.closed_loop.navigation import ObservationMap
from competition_user.closed_loop.timing import CommandHistory
from phase16z_s3_runtime.s3_timestamped_frames import PlanarPose
from phase17_berthing_perception.navigation_history import reproject_points

ROOT = Path(__file__).resolve().parents[1]


def test_future_command_cannot_be_inferred_as_past_disturbance():
    model = VesselDynamics.load(ROOT/"control_ws/src/competition_user/config/vessel_dynamics.yaml")
    history = CommandHistory()
    history.record(1.10, ThrustCommand(400., -400.))
    observer = DisturbanceEstimator(model)
    state = VesselState(0., 0., 0., 0., 0., 0.)
    observer.observe(state, 1., history)
    disturbance = observer.observe(state, 1.05, history)
    assert disturbance.yaw_moment_nm == pytest.approx(0.)
    assert disturbance.force_world_x_n == pytest.approx(0.)


def test_measurement_propagation_integrates_only_actual_command_intervals():
    model = VesselDynamics.load(ROOT/"control_ws/src/competition_user/config/vessel_dynamics.yaml")
    history = CommandHistory()
    history.record(1.05, ThrustCommand(80., 80.))
    history.record(1.15, ThrustCommand(-20., -20.))
    state = VesselState(0., 0., 0., 0., 0., 0.)
    from competition_user.closed_loop.dynamics import Disturbance
    actual = history.propagate(model, state, 1., 1.1, Disturbance())
    expected = state
    for dt in (.02, .02, .01):
        expected = model.step(expected, ThrustCommand(80., 80.), dt, Disturbance())
    assert actual.surge == pytest.approx(expected.surge)
    assert history.mean_command(1., 1.1).left_n == pytest.approx(40.)


def test_free_neighbor_does_not_clear_unknown_obstacle_subcell():
    observed = ObservationMap()
    observed.update(np.array([[100, -1]], dtype=np.int8), (0., 0., 0.), (0., 0., 0.), .2)
    observed.update(np.array([[-1, 0]], dtype=np.int8), (0., 0., 0.), (0., 0., 0.), .2)
    assert list(observed.occupied.values()) == [(0.1, 0.1)]
    observed.update(np.array([[0, -1]], dtype=np.int8), (0., 0., 0.), (0., 0., 0.), .2)
    assert not observed.occupied


def test_free_clearing_uses_current_grid_rotation_and_origin():
    observed = ObservationMap()
    observed.update(np.array([[100, -1]], dtype=np.int8), (0., 0., 0.), (0., 0., 0.), .2)
    # 船向东移动.2m，原障碍在新局部栅格的左邻格，不能被右格FREE清掉。
    observed.update(np.array([[-1, 0]], dtype=np.int8), (-.2, 0., 0.), (.2, 0., 0.), .2)
    assert observed.occupied


def test_async_clouds_reproject_the_same_fixed_object_with_roll_pitch_and_translation():
    first = PlanarPose(1_000_000_000, 1., 2., .3, .4, tuple(Rotation.from_euler("xyz", [.2, -.1, .3]).as_quat()))
    last = PlanarPose(1_350_000_000, 1.3, 2.1, .45, .5, tuple(Rotation.from_euler("xyz", [-.1, .2, .45]).as_quat()))
    world = np.array([[8., 3., 1.5], [9., -2., 2.1]])
    first_points = (world-np.array([1., 2., .4]))@Rotation.from_quat(first.quaternion_xyzw).as_matrix()
    expected = (world-np.array([1.3, 2.1, .5]))@Rotation.from_quat(last.quaternion_xyzw).as_matrix()
    assert reproject_points(first_points, first, last) == pytest.approx(expected)


def test_completion_requires_current_verified_safe_plan_without_extra_dwell():
    from competition_user.closed_loop.mission import Mission, MissionState, Bay
    mission = Mission()
    mission.bay = Bay(1, (-8., 0., 0.), (0., 0., 0.))
    mission.state = MissionState.FINAL
    state = VesselState(0., 0., 0., 0., 0., 0.)
    mission.check_completion(state, 10., False)
    assert mission.state == MissionState.FINAL
    # 官方按当前完成标志评分；本周期未通过安全检查时不能先盖章。
    mission.check_completion(state, 11., True)
    assert mission.state == MissionState.COMPLETE
