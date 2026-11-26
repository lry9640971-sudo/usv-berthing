"""导航无解时检查两段实际保持力；未证明安全的序列不参加回退。"""

from dataclasses import dataclass
import math
import time

import numpy as np

from .force_sequence import predict_sequence, shift_sequence
from .geometry import certify_swept_poses, moving_clearances


@dataclass(frozen=True)
class NavigationSequenceChoice:
    controls_n: np.ndarray
    durations_s: np.ndarray
    values: np.ndarray
    static_m: float
    dynamic_m: float
    stamp: float | None
    diagnostic: dict


class NavigationSequencePool:
    def __init__(self, model, hull, horizon=4., dt=.2):
        self.model, self.hull, self.horizon, self.dt = model, hull, horizon, dt
        self.previous = None
        self.last_diagnostics = {}
        low, high = model.actuator.allowed_input_range
        left = [model.actuator.left.force_from_input(value) for value in (low, 0., high)]
        right = [model.actuator.right.force_from_input(value) for value in (low, 0., high)]
        powers = np.asarray([(l, r) for l in left for r in right])
        pairs = np.asarray([(first, second) for first in powers for second in powers
                            if not np.array_equal(first, second)])
        # .8s与已有时序执行段相同，前后各两个预测步覆盖制动、转向的切换时机。
        holds = np.arange(2, 7, 2)*dt
        self.holds = holds[holds < horizon]
        self.controls = np.tile(pairs, (len(self.holds), 1, 1))
        first = np.repeat(self.holds, len(pairs))
        self.durations = np.column_stack((first, horizon-first))

    def invalidate(self):
        # 另一条指令已覆盖旧程序，不把未执行的旧轨迹作为连续动作。
        self.previous = None

    def _predict_many(self, state, disturbance):
        values = np.broadcast_to(tuple(state.__dict__.values()), (len(self.controls), 6)).copy()
        history = [values]
        for index in range(round(self.horizon/self.dt)):
            phase = (index*self.dt >= self.durations[:, 0]-1e-9).astype(int)
            commands = self.controls[np.arange(len(self.controls)), phase]
            for _ in range(2):
                values = self.model.step_batch(values, commands, self.dt/2., disturbance)
            history.append(values)
        return np.stack(history, axis=1)

    def choose(self, state, guidance, field, moving, disturbance, deadline, execution_snapshot=None):
        self.last_diagnostics = dict(sequence_candidates_evaluated=0, sequence_safe_candidates=0,
                                     sequence_certification_attempts=0, sequence_budget_reached=False)
        if time.perf_counter() >= deadline:
            self.last_diagnostics["sequence_budget_reached"] = True
            return None
        stamp = None
        if execution_snapshot is not None:
            state, moving, stamp = execution_snapshot()
        values = self._predict_many(state, disturbance)
        controls_list, durations_list = list(self.controls), list(self.durations)
        retained_list = [False]*len(values)

        def score(value, clearance):
            distance = max(0., float(np.linalg.norm(value[-1, :2]-guidance.goal_xy))-guidance.position_tolerance_m)
            # 与常规导航使用相同的目标进度及净空代价，没有新的完成或速度门。
            cost = 3.*distance + .7*abs(value[1, 3]-guidance.desired_surge)
            cost += .35*abs(value[1, 5]-guidance.desired_yaw_rate)
            if math.isfinite(clearance):
                cost += 1.2/max(.2, clearance+.3)
            return cost

        if self.previous is not None and stamp is not None:
            controls, durations, previous_stamp = self.previous
            shifted = shift_sequence(controls/100., durations, stamp-previous_stamp, self.horizon)
            if shifted is not None:
                controls, durations = shifted
                continued = predict_sequence(self.model, state, controls, durations, disturbance,
                    round(self.horizon/self.dt), self.dt)
                values = np.concatenate((values, continued[None]))
                controls_list.append(100.*controls)
                durations_list.append(durations)
                retained_list.append(True)
        # 先按目标进度的代价下界排序，再分批做完整几何。每批发现可行项就精查，
        # 不再等待216条粗检查全部结束后才给安全证明留时间。
        order = np.argsort([score(value, np.inf) for value in values], kind="stable")
        for begin in range(0, len(order), 8):
            if time.perf_counter() >= deadline:
                self.last_diagnostics["sequence_budget_reached"] = True
                break
            indices = order[begin:begin+8]
            poses = values[indices, :, :3]
            static = field.clearances(poses.reshape(-1, 3)).reshape(poses.shape[:2]).min(axis=1)
            dynamic = moving_clearances(poses, self.hull, moving, self.dt).min(axis=1)
            self.last_diagnostics["sequence_candidates_evaluated"] += len(indices)
            pending = [(score(values[index], min(static[local], dynamic[local])), index)
                for local, index in enumerate(indices) if min(static[local], dynamic[local]) >= 0.]
            self.last_diagnostics["sequence_safe_candidates"] += len(pending)
            for _, index in sorted(pending):
                if time.perf_counter() >= deadline:
                    self.last_diagnostics["sequence_budget_reached"] = True
                    break
                controls, durations, retained = controls_list[index], durations_list[index], retained_list[index]
                if execution_snapshot is not None:
                    state, moving, stamp = execution_snapshot()
                if retained:
                    old_controls, old_durations, old_stamp = self.previous
                    shifted = shift_sequence(old_controls/100., old_durations, stamp-old_stamp, self.horizon)
                    if shifted is None:
                        continue
                    controls, durations = shifted
                    controls = 100.*controls
                # 从实际新执行时刻重新积分每段力；粗采样可行不能直接标记SAFE。
                fine_dt = .02
                fine = predict_sequence(self.model, state, controls/100., durations, disturbance,
                                        round(self.horizon/fine_dt), fine_dt)
                certified, static_m, dynamic_m, diagnostic = certify_swept_poses(
                    fine[:, :3], field, moving, np.arange(len(fine))*fine_dt, deadline)
                self.last_diagnostics["sequence_certification_attempts"] += 1
                if not certified:
                    self.last_diagnostics["last_sequence_rejection"] = diagnostic
                    continue
                diagnostic.update(integration_dt_s=fine_dt)
                if stamp is not None:
                    self.previous = (controls.copy(), durations.copy(), stamp)
                return NavigationSequenceChoice(controls.copy(), durations.copy(), fine, static_m, dynamic_m,
                    stamp, dict(swept_certification=diagnostic, retained_force_sequence=retained))
        return None
