"""按实际命令的仿真时刻积分，测量与当前执行不再混用。"""

from collections import deque
import threading

from .actuator import ThrustCommand


class CommandHistory:
    def __init__(self):
        self.rows = deque([(-float("inf"), ThrustCommand(0., 0.))])
        self.lock = threading.Lock()

    def record(self, stamp, command):
        with self.lock:
            if stamp < self.rows[-1][0]:
                raise ValueError("COMMAND_TIME_REVERSED")
            if stamp == self.rows[-1][0]:
                self.rows[-1] = (stamp, command)
            elif command != self.rows[-1][1]:
                self.rows.append((stamp, command))
            # 保留窗口左端的命令，覆盖最多0.75s的已许可测量延迟。
            while len(self.rows) > 2 and self.rows[1][0] < stamp-30.:
                self.rows.popleft()

    def intervals(self, begin, end):
        if end < begin:
            raise ValueError("COMMAND_INTERVAL_REVERSED")
        command = ThrustCommand(0., 0.)
        cursor = begin
        # 执行线程可以新增真实命令；积分使用同一份不可变历史快照。
        with self.lock:
            rows = tuple(self.rows)
        for stamp, value in rows:
            if stamp <= begin:
                command = value
            elif stamp < end:
                yield cursor, stamp, command
                cursor, command = stamp, value
            else:
                break
        if end > cursor:
            yield cursor, end, command

    def mean_command(self, begin, end):
        left = right = 0.
        for first, last, command in self.intervals(begin, end):
            left += (last-first)*command.left_n
            right += (last-first)*command.right_n
        if end <= begin:
            return ThrustCommand(0., 0.)
        return ThrustCommand(left/(end-begin), right/(end-begin))

    def propagate(self, model, state, begin, end, disturbance):
        for first, last, command in self.intervals(begin, end):
            while last-first > 1e-9:
                dt = min(.02, last-first)
                state = model.step(state, command, dt, disturbance)
                first += dt
        return state
