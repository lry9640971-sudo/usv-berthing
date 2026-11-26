"""用真实ROS时钟消息复现控制求解阻塞，确认指令历史用实际发布时刻。"""

from pathlib import Path
import threading
import time

import pytest
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64

import competition_user.closed_loop.node as module
from competition_user.closed_loop.actuator import ThrustCommand
from competition_user.closed_loop.sim_clock import SimulationClockReader
from competition_user.closed_loop.force_sequence import TimedForceSequence
from competition_user.closed_loop.timing import CommandHistory

ROOT = Path(__file__).resolve().parents[1]


def test_command_timestamp_keeps_up_while_control_executor_is_busy(tmp_path, monkeypatch):
    directories = {'competition_user': ROOT/'control_ws/src/competition_user',
                   'phase17_berthing_perception': ROOT/'perception_ws/src/phase17_berthing_perception'}
    monkeypatch.setattr(module, 'get_package_share_directory', lambda package: str(directories[package]))
    rclpy.init(args=['--ros-args', '-p', 'runtime_output_dir:='+str(tmp_path)])
    source = Node('clock_test_source')
    publisher = source.create_publisher(Clock, '/clock', 10)
    reader = SimulationClockReader()
    node = module.ClosedLoopNode(simulation_clock=reader.clock)
    try:
        def publish_until(stamp):
            message = Clock()
            message.clock.sec = stamp
            deadline = time.monotonic()+3.
            while time.monotonic() < deadline:
                publisher.publish(message)
                if node.now_sim() == stamp:
                    return
                time.sleep(.01)
            raise AssertionError('独立仿真时钟没有接收到真实消息')

        publish_until(10)
        # 主控制执行器不spin，模拟数值求解占用；独立线程接收新的官方时钟。
        sender = threading.Thread(target=lambda: publish_until(11))
        sender.start()
        sender.join(timeout=3.5)
        assert not sender.is_alive()
        node.publish_command(ThrustCommand(10., -10.))
        assert node.last_command_stamp_sim_s == pytest.approx(11.)
        assert node.now_sim() == pytest.approx(11.)
    finally:
        node.execution_log.close()
        node.destroy_node()
        reader.close()
        source.destroy_node()
        rclpy.shutdown()


def test_short_force_segment_switches_on_ros_clock_without_spinning_planner(tmp_path,monkeypatch):
    directories={'competition_user':ROOT/'control_ws/src/competition_user',
                 'phase17_berthing_perception':ROOT/'perception_ws/src/phase17_berthing_perception'}
    monkeypatch.setattr(module,'get_package_share_directory',lambda p:str(directories[p]))
    rclpy.init(args=['--ros-args','-p','runtime_output_dir:='+str(tmp_path)])
    source=Node('force_clock_test_source')
    clock_publisher=source.create_publisher(Clock,'/clock',10)
    reader=SimulationClockReader()
    node=module.ClosedLoopNode(simulation_clock=reader.clock,force_timer_node=reader.node)
    received=[]
    reader.node.create_subscription(Float64,node.model.actuator.left.topic,
        lambda msg:received.append(msg.data),10)
    try:
        def advance(stamp,condition=lambda:True):
            message=Clock()
            message.clock.sec=int(stamp)
            message.clock.nanosec=round((stamp-int(stamp))*1e9)
            until=time.monotonic()+3.
            while time.monotonic()<until:
                clock_publisher.publish(message)
                if abs(node.now_sim()-stamp)<1e-8 and condition():
                    return
                time.sleep(.01)
            raise AssertionError('已批准推力没有按真实ROS时钟执行')
        advance(10.)
        node.force_executor.install(TimedForceSequence([[100.,0.],[-100.,0.],[50.,0.]],
            [.04,.8,.8],10.))
        # 主规划执行器始终不spin；40ms段不能被拉长到下一次152ms规划。
        advance(10.035,lambda:node.applied.left_n==100.)
        advance(10.045,lambda:node.applied.left_n==-100. and -100. in received)
        assert node.last_command_stamp_sim_s==pytest.approx(10.045)
        assert node.command_history.rows[-1][0]==pytest.approx(10.045)
        assert node.force_executor.transitions==1
        # 下一次规划覆盖旧序列；旧计划的后续段不得再恢复输出。
        node.publish_command(ThrustCommand(0.,0.))
        advance(10.90,lambda:node.applied.left_n==0.)
        assert node.applied==ThrustCommand(0.,0.)
    finally:
        node.execution_log.close()
        node.destroy_node()
        reader.close()
        source.destroy_node()
        rclpy.shutdown()


def test_history_interval_snapshot_survives_actual_command_recorded_during_integration():
    history=CommandHistory()
    history.record(0.,ThrustCommand(100.,0.))
    history.record(.5,ThrustCommand(-100.,0.))
    intervals=history.intervals(0.,1.)
    assert next(intervals)==(0.,.5,ThrustCommand(100.,0.))
    # 模拟独立执行回调恰在积分两段之间记入命令，不可混用半份新旧历史。
    history.record(.7,ThrustCommand(20.,0.))
    assert list(intervals)==[(.5,1.,ThrustCommand(-100.,0.))]
    assert list(history.intervals(0.,1.))[-1]==(.7,1.,ThrustCommand(20.,0.))
