"""真实执行单位、完整船模和制动动力学的反例回归。"""

from pathlib import Path

import pytest

from competition_user.closed_loop.actuator import Thruster, ThrustCommand
from competition_user.closed_loop.dynamics import VesselDynamics, VesselState, Disturbance, DisturbanceEstimator


CONFIG = Path(__file__).resolve().parents[1] / "control_ws/src/competition_user/config/vessel_dynamics.yaml"


def test_actual_force_mode_publishes_newtons_even_with_velocity_control():
    model = VesselDynamics.load(CONFIG)
    command = model.actuator.command(100., -80.)
    assert model.actuator.published_values(command) == (100., -80.)
    assert model.actuator.left.feedback_from_force(100.) == pytest.approx(118.88597930022942, rel=1e-6)
    assert model.actuator.command(800., -1500.) == ThrustCommand(400., -400.)
    assert model.actuator.left.min_command == -1000.


def test_explicit_angular_mode_uses_realizable_force_after_limit():
    thruster = Thruster("left", "/test", "rad/s", -100., 100., .004422, 1000., .2, (0., 1., 0.), (1., 0., 0.))
    assert thruster.input_from_force(100.) == 100.
    assert thruster.realizable_force(100.) == pytest.approx(70.752)
    assert thruster.feedback_from_force(100.) == pytest.approx(70.752)


def test_entire_robot_inertia_includes_engines_and_sensors():
    model = VesselDynamics.load(CONFIG)
    assert model.mass == pytest.approx(284.15575)
    assert model.inertia == pytest.approx(688.798775858)
    assert model.center[0] == pytest.approx(-.270260834)


def test_signed_turn_moment_and_drag_respect_actual_model():
    model = VesselDynamics.load(CONFIG)
    _, moment = model.force_moment(model.actuator.command(-100., 100.))
    assert moment == pytest.approx(205.427)
    state = VesselState(0., 0., 0., 1., 0., .2)
    drag = model.drag(state)
    assert drag[2] == pytest.approx(-192.)
    assert model.step(VesselState(0., 0., 0., 0., 0., 0.), ThrustCommand(-100., 100.), .1).yaw_rate > 0


def test_zero_force_does_not_cancel_sideways_drift():
    model = VesselDynamics.load(CONFIG)
    state = VesselState(0., 0., 0., 0., .3, 0.)
    stopped = model.predict(state, ThrustCommand(0., 0.), 3.)[-1]
    assert stopped.y > .45
    assert stopped.sway > 0


def test_reverse_force_reduces_actual_stopping_distance():
    model = VesselDynamics.load(CONFIG)
    state = VesselState(0., 0., 0., .8, 0., 0.)
    passive = model.predict(state, ThrustCommand(0., 0.), 2.)
    active = model.predict(state, ThrustCommand(-100., -100.), 2.)
    assert max(item.x for item in active) < .5 * max(item.x for item in passive)
    assert active[-1].surge < 0


def test_disturbance_estimate_uses_navigation_response_and_published_force():
    model = VesselDynamics.load(CONFIG)
    observer = DisturbanceEstimator(model, .5)
    state = VesselState(0., 0., 0., .2, .1, .05)
    command = ThrustCommand(80., 100.)
    actual = Disturbance(20., -15., 10.)
    for i in range(100):
        observer.observe(state, .05 * i, command)
        state = model.step(state, command, .05, actual)
    assert observer.estimate.force_world_x_n == pytest.approx(20., abs=.5)
    assert observer.estimate.force_world_y_n == pytest.approx(-15., abs=.5)
    assert observer.estimate.yaw_moment_nm == pytest.approx(10., abs=.5)
