"""终点滚动优化：共同规划减速与转向，直接约束真实船体和实际推力。"""

import math
import time

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .dynamics import VesselState
from .geometry import moving_clearances
from .force_sequence import shift_sequence, predict_sequence, uniform_initialization


class _SolverBudgetReached(Exception):
    """只中止数值迭代；已完成的物理检查和可执行候选保持有效。"""


class TerminalOptimizer:
    def __init__(self, model, hull, dt=.2, solve_budget_s=.25):
        self.model, self.hull, self.dt = model, hull, dt
        self.knots, self.steps_per_knot = 6, 4
        self.steps = self.knots*self.steps_per_knot
        self.approach_speed_mps = .75
        self.previous = None
        self.previous_sequence = None
        self.last_diagnostics = {}
        # 官方整个控制周期上限.5s，给末次安全重验及发布保留一半预算。
        # 迭代只改善已检查的候选，预算到期不允许省略完整安全检查。
        self.solve_budget_s = solve_budget_s

    def predict(self, state, controls, disturbance):
        controls = np.asarray(controls).reshape(-1, self.knots, 2)
        values = np.broadcast_to(tuple(state.__dict__.values()), (len(controls), 6)).copy()
        history = [values]
        for step in range(self.steps):
            force = 100.*controls[:, step//self.steps_per_knot]
            for _ in range(2):
                values = self.model.step_batch(values, force, self.dt/2, disturbance)
            history.append(values)
        return np.stack(history, axis=1)

    def seed(self, state, guidance, disturbance):
        if self.previous is not None:
            # 上次序列仅提供数值初值；每周期从新导航状态重新预测和验安全。
            shifted = self.previous.copy()
            shifted[:-1] = .75*self.previous[:-1]+.25*self.previous[1:]
            return shifted.ravel()
        return self.feedback_seed(state, guidance.desired_surge, guidance.desired_yaw_rate, disturbance)

    def feedback_seed(self, state, surge, yaw_rate, disturbance):
        controls = []
        for _ in range(self.knots):
            # 初值每段力保持.8s；不能借用实际.1s反馈时的.25s内环增益，
            # 否则无反馈的保持段会反复过冲，制造交替转向与不可行初值。
            command = self.model.speed_command(state, surge, yaw_rate, disturbance=disturbance,
                yaw_response_s=self.steps_per_knot*self.dt)
            controls.append((command.left_n/100., command.right_n/100.))
            for _ in range(self.steps_per_knot*2):
                state = self.model.step(state, command, self.dt/2, disturbance)
        return np.asarray(controls).ravel()

    def solve(self, state, guidance, field, moving, disturbance, dock_field=None, dock_penetration=0., execution_snapshot=None):
        started = time.perf_counter()
        deadline = None
        goal = np.asarray(guidance.goal_xy)
        c, s = math.cos(guidance.goal_yaw), math.sin(guidance.goal_yaw)
        axis, normal = np.asarray((c, s)), np.asarray((-s, c))
        start_along = float((np.asarray((state.x, state.y))-goal)@axis)
        times = np.arange(self.steps+1)*self.dt
        direction = -np.sign(start_along)
        reference_along = start_along+direction*np.minimum(abs(start_along), self.approach_speed_mps*times)
        reference_speed = direction*self.approach_speed_mps if abs(start_along) > self.approach_speed_mps*times[-1] else 0.
        candidates, cached, evaluations = [], {}, 0
        low, high = self.model.actuator.allowed_input_range
        force_low = np.asarray([item.force_from_input(low)/100. for item in
                                (self.model.actuator.left, self.model.actuator.right)])
        force_high = np.asarray([item.force_from_input(high)/100. for item in
                                 (self.model.actuator.left, self.model.actuator.right)])
        lower, upper = np.tile(force_low, self.knots), np.tile(force_high, self.knots)

        def objective(values, controls):
            delta = values[..., :2]-goal
            along, cross = delta@axis, delta@normal
            cs, sn = np.cos(values[..., 2]), np.sin(values[..., 2])
            vx = cs*values[..., 3]-sn*values[..., 4]
            vy = sn*values[..., 3]+cs*values[..., 4]
            velocity_along = c*vx+s*vy
            velocity_cross = -s*vx+c*vy
            # 位置和速度共用实际完成量纲；不把舒适距离排在标识点收敛之前。
            cost = np.mean((along-reference_along)**2+cross**2, axis=1)/.15**2
            cost += ((along[:, -1]-reference_along[-1])**2+cross[:, -1]**2)/.15**2
            cost += ((velocity_along[:, -1]-reference_speed)**2+velocity_cross[:, -1]**2)/.15**2
            cost += values[:, -1, 5]**2/.05**2
            # 姿态误差换算为船体端点位移，和标识点误差共用长度量纲；不是额外完成门。
            angle = np.arctan2(np.sin(values[:, -1, 2]-guidance.goal_yaw), np.cos(values[:, -1, 2]-guidance.goal_yaw))
            cost += (self.hull.radius*angle/.15)**2
            cost += .002*np.mean(controls**2, axis=1)
            return cost

        def clearances(poses):
            flat = poses.reshape(-1, 3)
            static = field.clearances(flat).reshape(len(poses), self.steps+1)
            if dock_field is not None:
                static = np.minimum(static, dock_field.clearances(flat).reshape(static.shape)+dock_penetration)
            dynamic = moving_clearances(poses, self.hull, moving, self.dt)
            return static, dynamic

        def evaluate_many(controls):
            nonlocal evaluations
            controls = np.atleast_2d(controls)
            values = self.predict(state, controls, disturbance)
            cost = objective(values, controls)
            static, dynamic = clearances(values[..., :3])
            constraints = np.minimum(static, dynamic)
            for i, control in enumerate(controls):
                item = (float(cost[i]), control.copy(), values[i], static[i], dynamic[i],
                        np.full(self.knots, self.steps_per_knot*self.dt), False)
                candidates.append(item)
                cached[control.tobytes()] = (cost[i], np.minimum(5., constraints[i]))
            evaluations += len(controls)
            return cost, np.minimum(5., constraints)

        derivatives = {}

        def value(control):
            if deadline is not None and time.perf_counter() >= deadline:
                raise _SolverBudgetReached()
            key = control.tobytes()
            if key not in cached:
                evaluate_many(control)
            return cached[key]

        def jacobians(control):
            if deadline is not None and time.perf_counter() >= deadline:
                raise _SolverBudgetReached()
            key = control.tobytes()
            if key not in derivatives:
                epsilon = 1e-4
                samples = np.repeat(control[None], len(control)+1, axis=0)
                difference = np.where(control+epsilon <= upper, epsilon, -epsilon)
                samples[1:] += np.diag(difference)
                predicted = self.predict(state, samples, disturbance)
                cost = objective(predicted, samples)
                # 几何只对当前轨迹的三个姿态变量求导，再与同模型受力敏感度组合。
                # 优化器每个实际试探序列仍由 value() 精查完整船体，导数不能伪造安全解。
                poses = np.repeat(predicted[0:1, :, :3], 4, axis=0)
                poses[1:] += (np.eye(3)*epsilon)[:, None, :]
                static, dynamic = clearances(poses)
                constraints = np.minimum(5., np.minimum(static, dynamic))
                geometry_gradient = ((constraints[1:]-constraints[0])/epsilon).T
                state_difference = predicted[1:, :, :3]-predicted[0, :, :3]
                state_difference[..., 2] = np.arctan2(np.sin(state_difference[..., 2]), np.cos(state_difference[..., 2]))
                sensitivities = state_difference/difference[:, None, None]
                derivatives[key] = ( (cost[1:]-cost[0])/difference,
                                     np.einsum('ptk,tk->tp', sensitivities, geometry_gradient) )
            return derivatives[key]

        seed = self.seed(state, guidance, disturbance)
        seed_source = "current_state_feedback"
        initial_seed_constraint = float(value(seed)[1].min())
        previous_seed_constraint = initial_seed_constraint if self.previous is not None else None
        feasible_seed_count = 0
        if initial_seed_constraint < 0.:
            # 初值也用当前观测精查；目标反馈及制动序列进入同一候选池。
            # 已可行的序列不会因优化失败丢失；只执行一次有墙钟预算的优化。
            seeds = np.stack((self.feedback_seed(state, guidance.desired_surge, guidance.desired_yaw_rate, disturbance),
                              self.feedback_seed(state, 0., 0., disturbance)))
            _, constraints = evaluate_many(seeds)
            feasible_seed_count = int(np.count_nonzero(constraints.min(axis=1) >= 0.))
            if not feasible_seed_count:
                # 反馈初值均不可行才检查实际推进器范围，不提升日常进场速度。
                left = (force_low[0], float(np.clip(0., force_low[0], force_high[0])), force_high[0])
                right = (force_low[1], float(np.clip(0., force_low[1], force_high[1])), force_high[1])
                powers = np.asarray([(l, r) for l in left for r in right])
                power_seeds = np.repeat(powers[:, None, :], self.knots, axis=1).reshape(len(powers), -1)
                _, power_constraints = evaluate_many(power_seeds)
                feasible_seed_count = int(np.count_nonzero(power_constraints.min(axis=1) >= 0.))
            # 优化从当前目标反馈开始；可行制动序列保留在 candidates 中。
            # 极值受力是安全候选，不拿饱和边界代替较易收敛的数值初值。
            seed = seeds[0]
        elif self.previous is not None:
            seed_source = "shifted_previous"
        bounds = list(zip(lower, upper))
        iterations = 0
        budget_reached = False
        def iteration_completed(_):
            nonlocal iterations
            iterations += 1
        deadline = None if self.solve_budget_s is None else started+self.solve_budget_s
        try:
            result = minimize(lambda x: value(x)[0], seed, jac=lambda x: jacobians(x)[0],
                              bounds=bounds, method="SLSQP",callback=iteration_completed,
                              constraints=[{"type": "ineq", "fun": lambda x: value(x)[1],
                                            "jac": lambda x: jacobians(x)[1]}],
                              # 官方门约束实际耗时；未到期时不因旧8次迭代门强行中断。
                              options={"maxiter": 100, "ftol": 1e-4})
            value(result.x)
        except _SolverBudgetReached:
            # 只接住本模块明确的计算截止信号；其它错误仍向外报告。
            budget_reached = True
            result = OptimizeResult(success=False,message="CALCULATION_BUDGET_REACHED",nit=iterations)
        deadline = None
        optimization_elapsed = time.perf_counter()-started
        realizable = [row for row in candidates if np.all(row[1] >= lower) and np.all(row[1] <= upper)]
        prediction_stamp = None
        continued_elapsed = None
        if execution_snapshot is not None:
            # 数值求解时旧推力仍在执行；同一候选池必须在实际新时刻重新验安全。
            # 不根据匿名停车规则覆盖命令，也不把假定延迟写成真实发布时间。
            state, moving, prediction_stamp = execution_snapshot()
            current_along = float((np.asarray((state.x,state.y))-goal)@axis)
            direction = -np.sign(current_along)
            reference_along = current_along+direction*np.minimum(abs(current_along), self.approach_speed_mps*times)
            reference_speed = direction*self.approach_speed_mps if abs(current_along) > self.approach_speed_mps*times[-1] else 0.
            unique = np.unique(np.stack([row[1] for row in realizable]),axis=0)
            # 先在同一当前状态算真实目标代价，再按代价依次精查。
            # 第一个可行者与全部重验后取最小代价等价；未检查者不声称SAFE。
            values = self.predict(state, unique, disturbance)
            costs = objective(values, unique)
            pending = [(float(cost), control, value, np.full(self.knots,self.steps_per_knot*self.dt), False)
                       for cost, control, value in zip(costs, unique, values)]
            if self.previous_sequence is not None:
                previous_controls, previous_durations, previous_stamp = self.previous_sequence
                continued_elapsed = prediction_stamp-previous_stamp
                shifted = shift_sequence(previous_controls,previous_durations,continued_elapsed,self.steps*self.dt)
                if shifted is not None:
                    controls, durations = shifted
                    value = predict_sequence(self.model,state,controls,durations,disturbance,self.steps,self.dt)
                    warm = uniform_initialization(controls,durations,self.knots,self.steps_per_knot*self.dt).ravel()
                    # 目标中的推力正则很小，但仍用真实保持时间加权，不能用平均力冒充轨迹。
                    cost = float(objective(value[None],warm[None])[0])
                    cost += .002*(np.sum(durations[:,None]*controls**2)/(2*self.steps*self.dt)-np.mean(warm**2))
                    pending.append((cost,controls.ravel(),value,durations,True))
            realizable = []
            for cost,control,value,durations,continued in sorted(pending,key=lambda item:item[0]):
                static,dynamic = clearances(value[None,:,:3])
                realizable.append((cost,control,value,static[0],dynamic[0],durations,continued))
                evaluations += 1
                if static.min()>=0. and dynamic.min()>=0.:
                    break
            if not any(row[3].min()>=0. and row[4].min()>=0. for row in realizable):
                # 新时刻反馈也进入同一候选池；只精查，不再做第二次迭代优化。
                candidates = []
                evaluate_many(np.stack((self.feedback_seed(state,guidance.desired_surge,guidance.desired_yaw_rate,disturbance),
                                        self.feedback_seed(state,0.,0.,disturbance))))
                realizable.extend(candidates)
        safe = [row for row in realizable if row[3].min() >= 0. and row[4].min() >= 0.]
        if safe:
            selected = min(safe, key=lambda row: row[0])
            status = "SAFE"
        else:
            # 数值失败或物理不可行均明确报告；没有伪造零速安全结果。
            selected = max(realizable, key=lambda row: min(row[3].min(), row[4].min()))
            status = "NO_SAFE_TRAJECTORY"
        _, controls, values, static, dynamic, durations, continued = selected
        controls = controls.reshape(-1, 2)
        self.previous = (uniform_initialization(controls,durations,self.knots,self.steps_per_knot*self.dt)
                         if continued else controls.copy())
        self.previous_sequence = (controls.copy(),durations.copy(),prediction_stamp) if prediction_stamp is not None else None
        command = self.model.actuator.command(*(100.*controls[0]))
        self.last_diagnostics = {"solver_success": bool(result.success), "solver_message": str(result.message),
                                 "solver_iterations": int(result.nit), "evaluated_sequences": evaluations,
                                 "certified_safe_sequences": len(safe), "decision": "direct_force_sequence",
                                  "seed_source": seed_source, "previous_seed_constraint_m": previous_seed_constraint,
                                  "feasible_seed_count": feasible_seed_count,
                                 "selected_controls_n": (100.*controls).tolist(),
                                 "selected_force_durations_s": durations.tolist(),
                                 "warm_start_controls_n": (100.*self.previous).tolist(),
                                 "retained_force_sequence": continued}
        if continued:
            self.last_diagnostics['continued_elapsed_s'] = continued_elapsed
        if prediction_stamp is not None:
            self.last_diagnostics['prediction_stamp_sim_s'] = prediction_stamp
            self.last_diagnostics['selection_at_execution_state'] = True
        self.last_diagnostics.update(solver_budget_reached=budget_reached,
            optimization_elapsed_s=optimization_elapsed,
            execution_selection_elapsed_s=time.perf_counter()-started-optimization_elapsed)
        trajectory = tuple(VesselState(*row) for row in values)
        return command, trajectory, status, float(static.min()), float(dynamic.min()), evaluations, len(safe)
