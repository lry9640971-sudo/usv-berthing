"""推力序列的真实保持时间；数值初值的平均值不代表可执行轨迹。"""

import numpy as np
from bisect import bisect_right
from itertools import accumulate

from .actuator import ThrustCommand


class TimedForceSequence:
    def __init__(self, controls_n, durations, start):
        self.commands = tuple(ThrustCommand(*control) for control in controls_n)
        self.start = float(start)
        ends = tuple(self.start+value for value in accumulate(durations))
        self.boundaries, self.end = ends[:-1], ends[-1]

    def command_at(self, stamp):
        return self.commands[bisect_right(self.boundaries, stamp)]

    def intervals(self, begin, end):
        cursor = begin
        for boundary in self.boundaries:
            if begin < boundary < end:
                yield cursor, boundary, self.command_at(cursor)
                cursor = boundary
        if end > cursor:
            yield cursor, end, self.command_at(cursor)


def shift_sequence(controls, durations, elapsed, horizon):
    if elapsed < 0. or elapsed >= sum(durations):
        return None
    remaining_controls, remaining_durations = [], []
    for control, duration in zip(controls, durations):
        used = min(elapsed, duration)
        elapsed -= used
        if duration-used > 1e-9:
            remaining_controls.append(control)
            remaining_durations.append(duration-used)
    # 延长末段也必须重新预测与检查，不能沿用上一周期的安全声明。
    remaining_durations[-1] += horizon-sum(remaining_durations)
    return np.asarray(remaining_controls), np.asarray(remaining_durations)


def predict_sequence(model, initial, controls, durations, disturbance, steps, dt):
    state, values = initial, [tuple(initial.__dict__.values())]
    index, elapsed, boundary = 0, 0., durations[0]
    for stop in np.arange(1, steps+1)*dt:
        while elapsed < stop-1e-9:
            if elapsed >= boundary-1e-9 and index+1 < len(controls):
                index += 1
                boundary += durations[index]
            interval = min(dt/2, stop-elapsed, boundary-elapsed)
            state = model.step(state, ThrustCommand(*(100.*controls[index])), interval, disturbance)
            elapsed += interval
        values.append(tuple(state.__dict__.values()))
    return np.asarray(values)


def uniform_initialization(controls, durations, knots, hold):
    boundaries = np.concatenate(([0.], np.cumsum(durations)))
    result = []
    for start in np.arange(knots)*hold:
        overlap = np.maximum(0., np.minimum(start+hold, boundaries[1:])-np.maximum(start, boundaries[:-1]))
        result.append(overlap@controls/hold)
    return np.asarray(result)
