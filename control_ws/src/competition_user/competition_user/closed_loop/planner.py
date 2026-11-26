"""唯一局部决策出口：同一实际受力模型选择并发布下一条推力。"""

from dataclasses import dataclass
import math
import time
from types import SimpleNamespace

import numpy as np

from .actuator import ThrustCommand
from .dynamics import Disturbance, VesselState, wrap
from .guidance import point_targets, terminal_surge
from .geometry import moving_clearances, swept_interval_bounds, certify_swept_poses
from .force_sequence import predict_sequence
from .terminal import TerminalOptimizer
from .navigation_sequences import NavigationSequencePool


@dataclass(frozen=True)
class MovingObstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius_m: float = 1.5
    velocity_std_mps: float = 0.


@dataclass(frozen=True)
class Guidance:
    goal_xy: tuple
    desired_surge: float
    desired_yaw_rate: float
    goal_yaw: float | None = None
    terminal: bool = False
    use_route: bool = True
    position_tolerance_m: float = 0.
    feedback: str = "constant"


@dataclass(frozen=True)
class Plan:
    command: ThrustCommand
    setpoint: tuple
    trajectory: tuple
    status: str
    minimum_static_clearance_m: float
    minimum_dynamic_clearance_m: float
    duration_s: float
    candidate_count: int
    safe_candidate_count: int


class ThrustPlanner:
    def __init__(self, dynamics, hull, horizon_s=4., dt=.2):
        self.model, self.hull = dynamics, hull
        self.horizon, self.dt = horizon_s, dt
        self.previous_setpoint = (0., 0.)
        self.terminal_optimizer = TerminalOptimizer(dynamics, hull, dt)
        self.navigation_sequences = NavigationSequencePool(dynamics, hull, horizon_s, dt)
        self.last_diagnostics = {}

    def _trajectory(self, initial, setpoint, disturbance, guidance=None):
        state, trajectory, commands = initial, [initial], []
        target_yaw = initial.yaw + setpoint[1]/.8
        for index in range(int(math.ceil(self.horizon / self.dt))):
            # 引导给的是航向误差反馈；预测也闭环收敛到该航向，不持续原地转四秒。
            yaw_rate = setpoint[1] if index == 0 else max(-.5, min(.5, .8*wrap(target_yaw-state.yaw)))
            surge = setpoint[0]
            if index > 0 and guidance is not None:
                if guidance.feedback == "point":
                    surge, yaw_rate = point_targets(state, guidance.goal_xy)
                elif guidance.feedback == "terminal":
                    surge = terminal_surge(state, guidance.goal_xy, guidance.goal_yaw)
                    surge *= max(0., math.cos(wrap(target_yaw-state.yaw)))
            command = self.model.speed_command(state, surge, yaw_rate, disturbance=disturbance)
            commands.append(command)
            # 每 0.1 s 积分的同一 RK2 模型；不把速度目标当成瞬时速度。
            for _ in range(2):
                state = self.model.step(state, command, self.dt / 2, disturbance)
            trajectory.append(state)
        return trajectory, commands

    def _dynamic_clearance(self, trajectory, obstacles):
        return moving_clearances([item.pose for item in trajectory], self.hull, obstacles, self.dt)

    def _actuation_surge_limits(self):
        low, high = self.model.actuator.allowed_input_range
        commands = (self.model.actuator.command(*(item.force_from_input(value) for item in
                    (self.model.actuator.left, self.model.actuator.right))) for value in (low, high))
        linear, quadratic = self.model.linear_drag[0], self.model.quadratic_drag[0]
        limits = []
        for command in commands:
            force, _ = self.model.force_moment(command)
            speed = (2*abs(force)/(linear+math.sqrt(linear*linear+4*quadratic*abs(force)))
                     if quadratic > 0 else abs(force)/linear)
            limits.append(math.copysign(speed, force))
        return limits

    def _constant_forces(self):
        low, high = self.model.actuator.allowed_input_range
        inputs = (low, float(np.clip(0., low, high)), high)
        left = [self.model.actuator.left.force_from_input(value) for value in inputs]
        right = [self.model.actuator.right.force_from_input(value) for value in inputs]
        return [self.model.actuator.command(l, r) for l in left for r in right]

    def _force_trajectory(self, state, command, disturbance):
        trajectory = [state]
        for _ in range(round(self.horizon / self.dt)):
            for _ in range(2):
                state = self.model.step(state, command, self.dt / 2., disturbance)
            trajectory.append(state)
        return trajectory, [command] * (len(trajectory) - 1)

    def _certify(self, state, trajectory, commands, field, moving, disturbance, static, dynamic, deadline):
        poses = np.asarray([item.pose for item in trajectory])
        times = np.arange(len(poses)) * self.dt
        bounds = swept_interval_bounds(poses, self.hull, moving, times, static, dynamic)
        if min(float(bounds[0].min()), float(bounds[1].min())) >= 0.:
            return True, trajectory, float(static.min()), float(dynamic.min()), {
                "reason": "SWEPT_INTERVAL_BOUND", "poses": len(poses), "refinements": 0,
                "static_lower_bound_m": float(bounds[0].min()), "dynamic_lower_bound_m": float(bounds[1].min())}
        # 接近边界时重积分实际保持的每段力，再精查间隙；不能仅加一个任意安全余量。
        fine_dt = .02
        values = predict_sequence(self.model, state,
            np.asarray([(item.left_n, item.right_n) for item in commands]) / 100.,
            np.full(len(commands), self.dt), disturbance, round(self.horizon / fine_dt), fine_dt)
        certified, minimum_static, minimum_dynamic, diagnostic = certify_swept_poses(
            values[:, :3], field, moving, np.arange(len(values)) * fine_dt, deadline)
        diagnostic["integration_dt_s"] = fine_dt
        trajectory = [VesselState(*value) for value in values[::round(self.dt / fine_dt)]]
        return certified, trajectory, minimum_static, minimum_dynamic, diagnostic

    def choose(self, state, guidance, static_field, moving=(), disturbance=Disturbance(),
               dock_field=None, permitted_dock_penetration_m=0., execution_snapshot=None):
        started = time.perf_counter()
        if guidance.feedback == "terminal":
            command, trajectory, status, static, dynamic, count, safe = self.terminal_optimizer.solve(
                state, guidance, static_field, moving, disturbance, dock_field, permitted_dock_penetration_m,
                execution_snapshot)
            self.last_diagnostics = self.terminal_optimizer.last_diagnostics
            return Plan(command, (trajectory[1].surge, trajectory[1].yaw_rate), trajectory,
                        status, static, dynamic, time.perf_counter()-started, count, safe)
        self.last_diagnostics = {"decision": "velocity_feedback_candidates", "emergency_candidates_evaluated": 0,
                                 "constant_force_candidates_evaluated": 0}
        desired_v, desired_w = guidance.desired_surge, guidance.desired_yaw_rate
        max_v = max(.35, abs(desired_v))
        surge = sorted(set((desired_v, 0., -.3, -.6, .2, .5 * max_v, max_v)))
        yaw = sorted(set((desired_w, 0., -.5, -.3, -.15, .15, .3, .5)))
        if guidance.terminal:
            surge = sorted(set((desired_v, 0., -.12, -.06, .06, .12, .24)))
            yaw = sorted(set((desired_w, 0., -.3, -.15, -.05, .05, .15, .3)))
        regular = [(v, w) for v in surge for w in yaw]
        low_v, high_v = self._actuation_surge_limits()
        emergency = [(v, w) for v in (low_v, low_v/2, high_v/2, high_v) for w in yaw if (v, w) not in regular]
        batches = [("feedback", regular)]
        if not guidance.terminal:
            batches.extend((("emergency_feedback", emergency), ("constant_force", self._constant_forces())))
        setpoints, trajectories, commands, static_rows, dynamic_rows, kinds = [], [], [], [], [], []
        recoveries, certificates = [], {}
        selected_index, prediction_stamp = None, None
        # 500ms是整个周期门；几何证明到400ms仍未完成就如实报告，给输入/发布留余量。
        certification_deadline = started + .4
        if dock_field is None:
            field = static_field
        else:
            field = SimpleNamespace(hull=self.hull, clearances=lambda poses: np.minimum(
                static_field.clearances(poses), dock_field.clearances(poses) + permitted_dock_penetration_m))
        goal = np.asarray(guidance.goal_xy)
        initial_distance = max(0., float(np.linalg.norm(goal - (state.x, state.y)))-guidance.position_tolerance_m)
        for kind, batch in batches:
            # 反馈速度族不能代表保持某个真实力；两者都从同一状态及受力模型预测。
            if kind == "constant_force":
                predicted, forces = zip(*(self._force_trajectory(state, command, disturbance) for command in batch))
                targets = [(row[1].surge, row[1].yaw_rate) for row in predicted]
                self.last_diagnostics["constant_force_candidates_evaluated"] = len(batch)
            else:
                predicted, forces = zip(*(self._trajectory(state, target, disturbance, guidance) for target in batch))
                targets = batch
            poses = np.asarray([[item.pose for item in trajectory] for trajectory in predicted])
            static = field.clearances(poses.reshape(-1, 3)).reshape(poses.shape[:2])
            dynamic = moving_clearances(poses, self.hull, moving, self.dt)
            offset = len(setpoints)
            setpoints.extend(targets)
            trajectories.extend(predicted)
            commands.extend(forces)
            static_rows.extend(static)
            dynamic_rows.extend(dynamic)
            kinds.extend([kind] * len(batch))
            if kind == "emergency_feedback":
                self.last_diagnostics["emergency_candidates_evaluated"] = len(batch)
            initial_static, initial_dynamic = float(static[0, 0]), float(dynamic[0, 0])
            safe = []
            for local_index, (trajectory, target) in enumerate(zip(predicted, targets)):
                index = offset+local_index
                minimum_static, minimum_dynamic = float(static[local_index].min()), float(dynamic[local_index].min())
                end = trajectory[-1]
                distance = max(0., float(np.linalg.norm(goal - (end.x, end.y)))-guidance.position_tolerance_m)
                progress = initial_distance - distance
                comfort = min(minimum_static, minimum_dynamic)
                endpoint_clearance = min(float(static[local_index, -1]), float(dynamic[local_index, -1]))
                cost = -3. * progress + .7 * abs(target[0] - desired_v) + .35 * abs(target[1] - desired_w)
                cost += .12 * abs(target[1] - self.previous_setpoint[1])
                cost += 1.2 / max(.2, comfort + .3) if math.isfinite(comfort) else 0.
                if guidance.goal_yaw is not None and not guidance.terminal:
                    cost += .5 * abs(wrap(end.yaw - guidance.goal_yaw))
                if guidance.terminal and distance < .5:
                    cost += 2. * end.speed
                if minimum_static >= 0. and minimum_dynamic >= 0.:
                    safe.append((cost, index))
                elif kind != "constant_force":
                    # 已进入观测边界时仍能脱离；零目标不是默认安全解。
                    no_deeper = minimum_static >= min(0., initial_static) - .015 and minimum_dynamic >= min(0., initial_dynamic) - .015
                    recovering = endpoint_clearance > min(initial_static, initial_dynamic) + .03
                    risk_cost = -5. * min(minimum_static, minimum_dynamic) - 3. * endpoint_clearance + .2 * end.speed
                    recoveries.append((0 if no_deeper and recovering else 1, risk_cost, index))
            if safe:
                if execution_snapshot is not None:
                    state, moving, prediction_stamp = execution_snapshot()
                for _, index in sorted(safe):
                    if execution_snapshot is not None:
                        trajectory, forces = (self._force_trajectory(state, commands[index][0], disturbance)
                            if kind == "constant_force" else self._trajectory(state, setpoints[index], disturbance, guidance))
                        commands[index] = forces
                        static_rows[index] = field.clearances([item.pose for item in trajectory])
                        dynamic_rows[index] = moving_clearances([item.pose for item in trajectory], self.hull, moving, self.dt)
                        trajectories[index] = trajectory
                    static, dynamic = static_rows[index], dynamic_rows[index]
                    if min(float(static.min()), float(dynamic.min())) < 0.:
                        certificates[index] = {"reason": "EXECUTION_STATE_COLLISION"}
                        certified = False
                    else:
                        certified, trajectory, minimum_static, minimum_dynamic, diagnostic = self._certify(
                            state, trajectories[index], commands[index], field, moving, disturbance,
                            static, dynamic, certification_deadline)
                        certificates[index] = diagnostic
                        trajectories[index] = trajectory
                        static_rows[index] = np.asarray([minimum_static])
                        dynamic_rows[index] = np.asarray([minimum_dynamic])
                    if certified:
                        selected_index = index
                        break
                    clearance = min(float(static_rows[index].min()), float(dynamic_rows[index].min()))
                    if kind != "constant_force":
                        recoveries.append((1, -5. * clearance, index))
                if selected_index is not None:
                    break
        if selected_index is None and not guidance.terminal:
            sequence = self.navigation_sequences.choose(state, guidance, field, moving, disturbance,
                certification_deadline, execution_snapshot)
            self.last_diagnostics.update(self.navigation_sequences.last_diagnostics)
            if sequence is not None:
                command = self.model.actuator.command(*sequence.controls_n[0])
                trajectory = tuple(VesselState(*value) for value in sequence.values[::round(self.dt/.02)])
                self.previous_setpoint = (trajectory[1].surge, trajectory[1].yaw_rate)
                self.last_diagnostics.update(sequence.diagnostic,
                    decision="navigation_force_sequence_escape",
                    selected_controls_n=sequence.controls_n.tolist(),
                    selected_force_durations_s=sequence.durations_s.tolist())
                if sequence.stamp is not None:
                    self.last_diagnostics.update(prediction_stamp_sim_s=sequence.stamp, selection_at_execution_state=True)
                return Plan(command, self.previous_setpoint, trajectory, "SAFE", sequence.static_m, sequence.dynamic_m,
                    time.perf_counter()-started, len(setpoints)+len(self.navigation_sequences.controls), 1)
        # 后续常规或无解命令会替换时序程序，旧序列不能继续充当已执行候选。
        self.navigation_sequences.invalidate()
        if selected_index is not None:
            index = selected_index
            status = "SAFE"
        else:
            rank, _, index = min(recoveries)
            status = "OBSERVED_BOUNDARY_SEPARATION" if rank == 0 else "NO_SAFE_TRAJECTORY"
            if execution_snapshot is not None:
                # 无解时也记录真正执行状态的受力和轨迹，不能保留旧姿态的脱离结论。
                state, moving, prediction_stamp = execution_snapshot()
                trajectory, forces = (self._force_trajectory(state, commands[index][0], disturbance)
                    if kinds[index] == "constant_force" else self._trajectory(state, setpoints[index], disturbance, guidance))
                trajectories[index], commands[index] = trajectory, forces
                static_rows[index] = field.clearances([item.pose for item in trajectory])
                dynamic_rows[index] = moving_clearances([item.pose for item in trajectory], self.hull, moving, self.dt)
                static, dynamic = static_rows[index], dynamic_rows[index]
                no_deeper = static.min() >= min(0., static[0]) - .015 and dynamic.min() >= min(0., dynamic[0]) - .015
                recovering = min(static[-1], dynamic[-1]) > min(static[0], dynamic[0]) + .03
                status = "OBSERVED_BOUNDARY_SEPARATION" if no_deeper and recovering else "NO_SAFE_TRAJECTORY"
        if kinds[index] == "constant_force":
            setpoints[index] = (trajectories[index][1].surge, trajectories[index][1].yaw_rate)
        self.previous_setpoint = setpoints[index]
        if kinds[index] == "emergency_feedback":
            self.last_diagnostics["decision"] = "actuation_range_escape"
        elif kinds[index] == "constant_force":
            self.last_diagnostics["decision"] = "constant_force_escape"
            self.last_diagnostics["constant_force_n"] = commands[index][0].__dict__
        if index in certificates:
            self.last_diagnostics["swept_certification"] = certificates[index]
        if prediction_stamp is not None:
            self.last_diagnostics.update(prediction_stamp_sim_s=prediction_stamp, selection_at_execution_state=True)
        return Plan(commands[index][0], setpoints[index], tuple(trajectories[index]), status,
                    float(static_rows[index].min()), float(dynamic_rows[index].min()), time.perf_counter() - started,
                    len(setpoints), int(selected_index is not None))
