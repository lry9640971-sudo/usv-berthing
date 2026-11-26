"""实际双推进器平面水动力；导航状态对应 base_link，受力对应质心。"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import yaml

from .actuator import Actuator, Thruster, ThrustCommand


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class VesselState:
    x: float
    y: float
    yaw: float
    surge: float
    sway: float
    yaw_rate: float

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    @property
    def speed(self):
        return math.hypot(self.surge, self.sway)


@dataclass(frozen=True)
class Disturbance:
    force_world_x_n: float = 0.
    force_world_y_n: float = 0.
    yaw_moment_nm: float = 0.


class VesselDynamics:
    def __init__(self, config):
        self.mass = float(config["mass_kg"])
        self.inertia = float(config["yaw_inertia_com_kg_m2"])
        self.center = tuple(config["center_of_mass_base_m"][:2])
        self.linear_drag = tuple(config["linear_drag"][key] for key in ("surge", "sway", "yaw"))
        self.quadratic_drag = tuple(config["quadratic_drag"][key] for key in ("surge", "sway", "yaw"))
        self.actuator = Actuator([Thruster.from_config(row) for row in config["thrusters"]], config["allowed_input_range"])
        if self.mass <= 0 or self.inertia <= 0:
            raise ValueError("船模质量或惯量无效")
        if any(not np.allclose(item.axis_base[:2], (1., 0.), atol=1e-8)
               for item in (self.actuator.left, self.actuator.right)):
            raise ValueError("当前控制只支持原始零转向角的双纵向推进器")

    @classmethod
    def load(cls, path):
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def body_com_velocity(self, state):
        cx, cy = self.center
        return state.surge - state.yaw_rate * cy, state.sway + state.yaw_rate * cx

    def com_world_velocity(self, state):
        u, v = self.body_com_velocity(state)
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        return np.asarray((c * u - s * v, s * u + c * v))

    def force_moment(self, command):
        left, right = self.actuator.left, self.actuator.right
        moment = -(left.position_base_m[1] - self.center[1]) * command.left_n
        moment -= (right.position_base_m[1] - self.center[1]) * command.right_n
        return command.left_n + command.right_n, moment

    def drag(self, state):
        velocity = (*self.body_com_velocity(state), state.yaw_rate)
        return tuple(-(linear + quadratic * abs(value)) * value
                     for value, linear, quadratic in zip(velocity, self.linear_drag, self.quadratic_drag))

    def derivative(self, state, command, disturbance=Disturbance()):
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        u, v = self.body_com_velocity(state)
        thrust, moment = self.force_moment(command)
        dx, dy, dr = self.drag(state)
        force_x = c * disturbance.force_world_x_n + s * disturbance.force_world_y_n
        force_y = -s * disturbance.force_world_x_n + c * disturbance.force_world_y_n
        angular_accel = (moment + dr + disturbance.yaw_moment_nm) / self.inertia
        du = (thrust + dx + force_x) / self.mass + state.yaw_rate * v
        dv = (dy + force_y) / self.mass - state.yaw_rate * u
        cx, cy = self.center
        return (c * state.surge - s * state.sway, s * state.surge + c * state.sway, state.yaw_rate,
                du + angular_accel * cy, dv - angular_accel * cx, angular_accel)

    def step(self, state, command, dt, disturbance=Disturbance()):
        values = tuple(state.__dict__.values())
        first = self.derivative(state, command, disturbance)
        middle = VesselState(*(value + rate * dt / 2 for value, rate in zip(values, first)))
        rates = self.derivative(middle, command, disturbance)
        result = [value + rate * dt for value, rate in zip(values, rates)]
        result[2] = wrap(result[2])
        return VesselState(*result)

    def step_batch(self, values, forces, dt, disturbance=Disturbance()):
        """同一 RK2 受力模型的批量数值接口，供优化器同时计算扰动导数。"""
        values, forces = np.asarray(values, dtype=float), np.asarray(forces, dtype=float)

        def rates(state):
            x, y, yaw, surge, sway, yaw_rate = np.moveaxis(state, -1, 0)
            c, s = np.cos(yaw), np.sin(yaw)
            cx, cy = self.center
            u, v = surge-yaw_rate*cy, sway+yaw_rate*cx
            dx = -(self.linear_drag[0]+self.quadratic_drag[0]*np.abs(u))*u
            dy = -(self.linear_drag[1]+self.quadratic_drag[1]*np.abs(v))*v
            dr = -(self.linear_drag[2]+self.quadratic_drag[2]*np.abs(yaw_rate))*yaw_rate
            yl = self.actuator.left.position_base_m[1]-cy
            yr = self.actuator.right.position_base_m[1]-cy
            moment = -yl*forces[..., 0]-yr*forces[..., 1]
            angular = (moment+dr+disturbance.yaw_moment_nm)/self.inertia
            fx = c*disturbance.force_world_x_n+s*disturbance.force_world_y_n
            fy = -s*disturbance.force_world_x_n+c*disturbance.force_world_y_n
            du = (forces.sum(axis=-1)+dx+fx)/self.mass+yaw_rate*v
            dv = (dy+fy)/self.mass-yaw_rate*u
            return np.stack((c*surge-s*sway, s*surge+c*sway, yaw_rate,
                             du+angular*cy, dv-angular*cx, angular), axis=-1)

        middle = values+rates(values)*dt/2
        result = values+rates(middle)*dt
        result[..., 2] = np.arctan2(np.sin(result[..., 2]), np.cos(result[..., 2]))
        return result

    def predict(self, state, command, horizon, dt=.1, disturbance=Disturbance()):
        trajectory = [state]
        for _ in range(int(math.ceil(horizon / dt))):
            state = self.step(state, command, dt, disturbance)
            trajectory.append(state)
        return trajectory

    def speed_command(self, state, target_surge, target_yaw_rate, disturbance=Disturbance(), response_s=1., yaw_response_s=.25):
        # 水动力前馈与速度误差共用模型；最终分配后返回的就是将发布的力。
        dx, _, dr = self.drag(state)
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        external_x = c * disturbance.force_world_x_n + s * disturbance.force_world_y_n
        total = self.mass * (target_surge - state.surge) / response_s - dx - external_x
        # 外环航向增益 .8；角速度内环 .25 s 满足临界阻尼条件，避免旧 1 s 内环过冲。
        moment = self.inertia * (target_yaw_rate - state.yaw_rate) / yaw_response_s - dr - disturbance.yaw_moment_nm
        yl = self.actuator.left.position_base_m[1] - self.center[1]
        yr = self.actuator.right.position_base_m[1] - self.center[1]
        left = (-moment - yr * total) / (yl - yr)
        right = total - left
        return self.actuator.command(left, right)


class DisturbanceEstimator:
    """使用生产导航反馈估计未建模受力；不读取仿真真值。"""

    def __init__(self, model, time_constant_s=2.):
        self.model = model
        self.time_constant = time_constant_s
        self.estimate = Disturbance()
        self.previous = None

    def observe(self, state, stamp, applied):
        if self.previous is not None:
            old, previous_stamp = self.previous
            dt = stamp - previous_stamp
            if .01 <= dt <= .25:
                acceleration = (self.model.com_world_velocity(state) - self.model.com_world_velocity(old)) / dt
                mean = VesselState(*(a + (b - a) / 2 for a, b in zip(old.__dict__.values(), state.__dict__.values())))
                mean = VesselState(mean.x, mean.y, old.yaw + wrap(state.yaw - old.yaw) / 2,
                                   mean.surge, mean.sway, mean.yaw_rate)
                # 观测区间只能由当时已经施加的命令解释。
                command = applied.mean_command(previous_stamp, stamp) if hasattr(applied, "mean_command") else applied
                thrust, moment = self.model.force_moment(command)
                dx, dy, dr = self.model.drag(mean)
                c, s = math.cos(mean.yaw), math.sin(mean.yaw)
                residual = (self.model.mass * acceleration[0] - c * (thrust + dx) + s * dy,
                            self.model.mass * acceleration[1] - s * (thrust + dx) - c * dy,
                            self.model.inertia * (state.yaw_rate - old.yaw_rate) / dt - moment - dr)
                alpha = 1 - math.exp(-dt / self.time_constant)
                self.estimate = Disturbance(*(a + alpha * (b - a) for a, b in zip(self.estimate.__dict__.values(), residual)))
        self.previous = state, stamp
        return self.estimate
