"""执行同一规划出口批准的推力时序；不在执行层另选避障或停车策略。"""

from bisect import bisect_right
import threading
import time

from .actuator import ThrustCommand


class ApprovedForceExecutor:
    def __init__(self, now, publish):
        self.now, self.publish = now, publish
        self.lock = threading.RLock()
        self.sequence = None
        self.index = 0
        self.command = ThrustCommand(0., 0.)
        self.stamp = None
        self.closed = False
        self.max_cycle_s = self.max_lateness_s = 0.
        self.transitions = 0

    def _emit(self, command, planned_stamp, source):
        self.stamp = self.publish(command, planned_stamp, source)
        self.command = command
        self.max_lateness_s = max(self.max_lateness_s, self.stamp-planned_stamp)

    def replace_constant(self, command):
        started = time.perf_counter()
        with self.lock:
            self.sequence = None
            self._emit(command, self.now(), 'planner_command')
        self.max_cycle_s = max(self.max_cycle_s, time.perf_counter()-started)

    def install(self, sequence):
        started = time.perf_counter()
        with self.lock:
            self.sequence = sequence
            now = self.now()
            self.index = bisect_right(sequence.boundaries, now)
            planned = sequence.start if not self.index else sequence.boundaries[self.index-1]
            self._emit(sequence.command_at(now), planned, 'planner_sequence')
        self.max_cycle_s = max(self.max_cycle_s, time.perf_counter()-started)

    def tick(self):
        started = time.perf_counter()
        with self.lock:
            if self.closed or self.sequence is None:
                return
            now = self.now()
            index = bisect_right(self.sequence.boundaries, now)
            if index != self.index:
                self.index = index
                command = self.sequence.command_at(now)
                if command != self.command:
                    self._emit(command, self.sequence.boundaries[index-1], 'approved_sequence_transition')
                    self.transitions += 1
        self.max_cycle_s = max(self.max_cycle_s, time.perf_counter()-started)

    def snapshot(self):
        with self.lock:
            return self.stamp, self.command

    def close(self):
        with self.lock:
            self.sequence = None
            self.closed = True
