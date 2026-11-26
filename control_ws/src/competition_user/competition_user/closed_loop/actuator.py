"""推进器唯一单位转换边界；规划始终使用实际牛顿值。"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Thruster:
    side: str
    topic: str
    input_unit: str
    min_command: float
    max_command: float
    coefficient: float
    fluid_density_kg_m3: float
    diameter_m: float
    position_base_m: tuple
    axis_base: tuple

    @classmethod
    def from_config(cls, values):
        item = cls(**{**values, "position_base_m": tuple(values["position_base_m"]),
                      "axis_base": tuple(values["axis_base"])})
        if item.input_unit not in ("N", "rad/s") or item.coefficient <= 0 or item.min_command >= item.max_command:
            raise ValueError("推进器配置无效或未实现")
        return item

    @property
    def force_coefficient(self):
        return self.coefficient * self.fluid_density_kg_m3 * self.diameter_m ** 4

    def force_from_input(self, command):
        if not math.isfinite(command):
            raise ValueError("推进器输入不是有限值")
        clipped = min(self.max_command, max(self.min_command, command))
        return clipped if self.input_unit == "N" else self.force_coefficient * abs(clipped) * clipped

    def input_from_force(self, force):
        if not math.isfinite(force):
            raise ValueError("请求推力不是有限值")
        value = force if self.input_unit == "N" else math.copysign(math.sqrt(abs(force) / self.force_coefficient), force)
        return min(self.max_command, max(self.min_command, value))

    def realizable_force(self, requested):
        return self.force_from_input(self.input_from_force(requested))

    def feedback_from_force(self, force):
        # ForceCmd 的反馈为 rad/s；AngVelCmd 的反馈为 N。
        actual = self.realizable_force(force)
        return math.copysign(math.sqrt(abs(actual) / self.force_coefficient), actual) if self.input_unit == "N" else actual


@dataclass(frozen=True)
class ThrustCommand:
    left_n: float
    right_n: float


class Actuator:
    def __init__(self, thrusters, allowed_input_range):
        self.left = next(item for item in thrusters if item.side == "left")
        self.right = next(item for item in thrusters if item.side == "right")
        self.allowed_input_range = tuple(allowed_input_range)

    def command(self, left_n, right_n):
        def admissible(item, force):
            value = item.input_from_force(force)
            value = min(self.allowed_input_range[1], max(self.allowed_input_range[0], value))
            return item.force_from_input(value)
        return ThrustCommand(admissible(self.left, left_n), admissible(self.right, right_n))

    def published_values(self, command):
        return (self.left.input_from_force(command.left_n), self.right.input_from_force(command.right_n))
