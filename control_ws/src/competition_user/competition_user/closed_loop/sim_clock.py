"""独立接收官方仿真时钟，已批准的推力时序不被规划求解阻塞。"""

import threading

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter


class SimulationClockReader:
    def __init__(self):
        self.node = Node('closed_loop_clock', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.clock = self.node.get_clock()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        # 本线程处理时钟及已批准力的切换；业务输入、任务状态和求解仍顺序处理。
        self.thread = threading.Thread(target=self.executor.spin, name='closed-loop-clock', daemon=True)
        self.thread.start()

    def close(self):
        self.executor.shutdown(timeout_sec=2.)
        self.thread.join(timeout=2.)
        self.executor.remove_node(self.node)
        self.node.destroy_node()
