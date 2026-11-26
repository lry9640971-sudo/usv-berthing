"""终点优化必须预测真实发布的力，覆盖减速中的侧移反例。"""

from pathlib import Path
import json
from dataclasses import replace

import numpy as np
import pytest

from competition_user.closed_loop.actuator import ThrustCommand
from competition_user.closed_loop.dynamics import Disturbance, VesselDynamics, VesselState
from competition_user.closed_loop.geometry import CollisionField, Hull, moving_clearances, transform_xy
from competition_user.closed_loop.planner import Guidance, ThrustPlanner, MovingObstacle


ROOT = Path(__file__).resolve().parents[1]


def model_and_hull():
    return (VesselDynamics.load(ROOT/'control_ws/src/competition_user/config/vessel_dynamics.yaml'),
            Hull.load(ROOT/'perception_ws/src/phase17_berthing_perception/config/hull_collision_footprint_base_link.yaml'))


def test_batched_rk2_matches_actual_force_model_for_turns_and_drift():
    model, _ = model_and_hull()
    random = np.random.default_rng(27)
    states = random.uniform([-10., -10., -3., -.8, -.3, -.4], [10., 10., 3., .8, .3, .4], (40, 6))
    forces = random.uniform(-400., 400., (40, 2))
    disturbance = Disturbance(10., 5., 2.)
    reference = np.asarray([tuple(model.step(VesselState(*state), ThrustCommand(*force), .1, disturbance).__dict__.values())
                            for state, force in zip(states, forces)])
    assert model.step_batch(states, forces, .1, disturbance) == pytest.approx(reference, abs=1e-12)


def test_terminal_first_published_force_is_exactly_the_predicted_force():
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    state = VesselState(.65, 3.07, -.024, .259, .033, 0.)
    disturbance = Disturbance(10., 5., 2.)
    guidance = Guidance((1.5, 3.), .18, -.23, 0., True, feedback='terminal')
    plan = planner.choose(state, guidance, CollisionField([], hull), disturbance=disturbance)
    expected = model.step(model.step(state, plan.command, .1, disturbance), plan.command, .1, disturbance)
    assert tuple(plan.trajectory[1].__dict__.values()) == pytest.approx(tuple(expected.__dict__.values()), abs=1e-12)
    assert model.actuator.published_values(plan.command) == (plan.command.left_n, plan.command.right_n)
    assert all(-400. <= value <= 400. for value in plan.command.__dict__.values())
    assert abs(plan.trajectory[-1].y-3.) < abs(state.y-3.)
    assert plan.status == 'SAFE'


def recorded_terminal_case(suffix='465904'):
    fixture = json.loads((ROOT/f'test/fixtures/terminal_{suffix}.json').read_text(encoding='utf-8'))
    data = np.load(ROOT/f'test/fixtures/terminal_{suffix}.npz', allow_pickle=False)
    model, hull = model_and_hull()
    planner = ThrustPlanner(model, hull)
    planner.previous_setpoint = tuple(fixture['previous_setpoint'])
    planner.terminal_optimizer.previous = np.asarray(fixture['previous_controls_n'])/100.
    state = VesselState(**fixture['vessel'])
    guidance = Guidance(**fixture['guidance'])
    disturbance = Disturbance(**fixture['disturbance'])
    moving = [MovingObstacle(**{k:v for k,v in row.items() if k != 'track_id'}) for row in fixture['moving_obstacles']]
    field = CollisionField(data['points'], hull, float(data['point_radius_m']))
    return fixture, planner, state, guidance, disturbance, moving, field


@pytest.mark.parametrize('suffix', ('465904', '465688', '465800'))
def test_recorded_infeasible_old_sequence_does_not_hide_real_current_safe_solution(suffix):
    fixture, planner, state, guidance, disturbance, moving, field = recorded_terminal_case(suffix)
    seed = planner.terminal_optimizer.seed(state, guidance, disturbance)
    old_poses = planner.terminal_optimizer.predict(state, seed, disturbance)[0, :, :3]
    assert field.clearances(old_poses).min() < 0.
    plan = planner.choose(state, guidance, field, moving, disturbance)
    assert plan.status == 'SAFE'
    poses = np.asarray([item.pose for item in plan.trajectory])
    assert field.clearances(poses).min() >= 0.
    assert moving_clearances(poses, field.hull, moving, .2).min() >= 0.
    assert max(abs(plan.command.left_n), abs(plan.command.right_n)) <= 400.


def test_reinitialization_cannot_claim_safe_when_current_full_hull_is_blocked():
    _, planner, state, guidance, disturbance, moving, field = recorded_terminal_case()
    # 人为添加当前真实部件内部的障碍；任何未来转向都不能掩盖起点碰撞。
    inside_local = np.mean(field.hull.components[0], axis=0)
    inside_world = transform_xy([inside_local], state.pose)
    blocked = CollisionField(np.concatenate((field.points, inside_world)), field.hull, field.point_radius)
    plan = planner.choose(state, guidance, blocked, moving, disturbance)
    assert plan.status == 'NO_SAFE_TRAJECTORY'
    assert plan.minimum_static_clearance_m < 0.


@pytest.mark.parametrize('rate',(-.18,.18))
def test_terminal_braking_seed_does_not_oscillate_between_long_force_holds(rate):
    model,hull=model_and_hull()
    optimizer=ThrustPlanner(model,hull).terminal_optimizer
    state=VesselState(0.,0.,-.3,.2,.1,rate)
    disturbance=Disturbance(3.,17.,-12.)
    seed=optimizer.feedback_seed(state,0.,0.,disturbance)
    predicted=optimizer.predict(state,seed,disturbance)[0]
    # 与真实.8s保持一致的初值角速度应逐步衰减，不能像旧.25s设计那样反复过冲。
    yaw_rate=predicted[::optimizer.steps_per_knot,5]*np.sign(rate)
    assert np.all(yaw_rate>=0.)
    assert np.all(np.diff(yaw_rate)<0.)
    assert yaw_rate[-1]<.01


def test_saved_clock_delay_revalidates_same_candidates_at_real_execution_state():
    fixture=json.loads((ROOT/'test/fixtures/terminal_execution_11600.json').read_text(encoding='utf-8'))
    data=np.load(ROOT/'test/fixtures/terminal_execution_11600.npz',allow_pickle=False)
    model,hull=model_and_hull()
    field=CollisionField(data['points'],hull,float(data['point_radius_m']))
    planner=ThrustPlanner(model,hull)
    planner.terminal_optimizer.previous=np.asarray(fixture['previous_controls_n'])/100.
    old=VesselState(**fixture['vessel'])
    current=VesselState(**fixture['execution_vessel'])
    disturbance=Disturbance(**fixture['disturbance'])
    guidance=Guidance(**fixture['guidance'])
    moving=[MovingObstacle(**{k:v for k,v in r.items() if k!='track_id'}) for r in fixture['moving_obstacles']]
    delay=fixture['execution_stamp_sim_s']-fixture['stamp_sim_s']
    advanced=[replace(o,x=o.x+o.vx*delay,y=o.y+o.vy*delay) for o in moving]
    # 原记录求解前称SAFE；把原序列放到真正发布状态后会侵入4厘米。
    original=np.asarray(fixture['decision_diagnostics']['selected_controls_n'])/100.
    late=planner.terminal_optimizer.predict(current,original,disturbance)[0]
    assert field.clearances(late[:,:3]).min()<-.03
    plan=planner.choose(old,guidance,field,moving,disturbance,
        execution_snapshot=lambda:(current,advanced,fixture['execution_stamp_sim_s']))
    assert plan.status=='SAFE'
    assert plan.trajectory[0]==current
    assert field.clearances([s.pose for s in plan.trajectory]).min()>=0.
    assert moving_clearances([s.pose for s in plan.trajectory],hull,advanced,.2).min()>=0.
    assert planner.last_diagnostics['prediction_stamp_sim_s']==pytest.approx(fixture['execution_stamp_sim_s'])
    assert model.actuator.published_values(plan.command)==(plan.command.left_n,plan.command.right_n)


@pytest.mark.parametrize('blocked',(False,True))
def test_solver_deadline_keeps_physical_certification_for_recorded_full_run(blocked):
    fixture=json.loads((ROOT/'test/fixtures/terminal_budget_462448.json').read_text(encoding='utf-8'))
    data=np.load(ROOT/'test/fixtures/terminal_budget_462448.npz',allow_pickle=False)
    model,hull=model_and_hull()
    old=VesselState(**fixture['vessel'])
    current=VesselState(**fixture['prediction_initial_vessel'])
    points=data['points']
    if blocked:
        inside=transform_xy([np.mean(hull.components[0],axis=0)],current.pose)
        points=np.concatenate((points,inside))
    field=CollisionField(points,hull,float(data['point_radius_m']))
    planner=ThrustPlanner(model,hull)
    planner.terminal_optimizer.previous=np.asarray(fixture['previous_controls_n'])/100.
    # 明确用零迭代预算复现截止；初值和执行状态的完整检查仍必须运行。
    planner.terminal_optimizer.solve_budget_s=0.
    plan=planner.choose(old,Guidance(**fixture['guidance']),field,
        disturbance=Disturbance(**fixture['disturbance']),
        execution_snapshot=lambda:(current,[],fixture['prediction_stamp_sim_s']))
    assert planner.last_diagnostics['solver_budget_reached']
    assert planner.last_diagnostics['solver_iterations']==0
    assert plan.trajectory[0]==current
    actual=field.clearances([s.pose for s in plan.trajectory])
    if blocked:
        assert plan.status=='NO_SAFE_TRAJECTORY'
        assert actual.min()<0.
    else:
        assert plan.status=='SAFE'
        assert actual.min()>=0.
        assert all(abs(v)<=400. for v in plan.command.__dict__.values())


def recorded_timed_continuation_case():
    fixture=json.loads((ROOT/'test/fixtures/terminal_retention_2200.json').read_text(encoding='utf-8'))
    data=np.load(ROOT/'test/fixtures/terminal_budget_462448.npz',allow_pickle=False)
    model,hull=model_and_hull()
    planner=ThrustPlanner(model,hull)
    optimizer=planner.terminal_optimizer
    controls=np.asarray(fixture['previous_controls_n'])/100.
    optimizer.previous=controls.copy()
    optimizer.previous_sequence=(controls,np.asarray(fixture['previous_force_durations_s']),
        fixture['previous_prediction_stamp_sim_s'])
    optimizer.solve_budget_s=0.
    field=CollisionField(data['points'],hull,float(data['point_radius_m']))
    return fixture,planner,field


def test_true_force_timing_preserves_safe_remaining_path_that_averaging_loses():
    fixture,planner,field=recorded_timed_continuation_case()
    old=VesselState(**fixture['vessel'])
    current=VesselState(**fixture['prediction_initial_vessel'])
    disturbance=Disturbance(**fixture['disturbance'])
    guidance=Guidance(**fixture['guidance'])
    optimizer=planner.terminal_optimizer
    mixed=optimizer.predict(current,optimizer.seed(current,guidance,disturbance),disturbance)[0]
    assert field.clearances(mixed[:,:3]).min()<0.
    plan=planner.choose(old,guidance,field,disturbance=disturbance,
        execution_snapshot=lambda:(current,[],fixture['prediction_stamp_sim_s']))
    assert plan.status=='SAFE'
    assert planner.last_diagnostics['retained_force_sequence']
    assert planner.last_diagnostics['selected_force_durations_s']==pytest.approx([.6,.8,.8,.8,.8,1.])
    assert plan.command==ThrustCommand(*fixture['previous_controls_n'][0])
    assert field.clearances([s.pose for s in plan.trajectory]).min()>.02
    # 首段真实保持.6s，然后切下一段力；不能执行两个力的平均值。
    expected=current
    for _ in range(6):
        expected=planner.model.step(expected,plan.command,.1,disturbance)
    assert tuple(plan.trajectory[3].__dict__.values())==pytest.approx(tuple(expected.__dict__.values()),abs=1e-11)
    next_force=ThrustCommand(*fixture['previous_controls_n'][1])
    for _ in range(2):
        expected=planner.model.step(expected,next_force,.1,disturbance)
    assert tuple(plan.trajectory[4].__dict__.values())==pytest.approx(tuple(expected.__dict__.values()),abs=1e-11)


def test_saved_force_sequence_cannot_reuse_safety_after_new_physical_obstacle():
    fixture,planner,field=recorded_timed_continuation_case()
    current=VesselState(**fixture['prediction_initial_vessel'])
    inside=transform_xy([np.mean(field.hull.components[0],axis=0)],current.pose)
    blocked=CollisionField(np.concatenate((field.points,inside)),field.hull,field.point_radius)
    plan=planner.choose(VesselState(**fixture['vessel']),Guidance(**fixture['guidance']),blocked,
        disturbance=Disturbance(**fixture['disturbance']),
        execution_snapshot=lambda:(current,[],fixture['prediction_stamp_sim_s']))
    assert plan.status=='NO_SAFE_TRAJECTORY'
    assert plan.minimum_static_clearance_m<0.
