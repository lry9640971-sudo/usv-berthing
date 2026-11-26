"""独立短时执行诊断：通过生产导航和推进器反馈核对单位、预测及制动。"""

from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import sys
import signal
from threading import Lock
import time

from ament_index_python.packages import get_package_share_directory
import gz.transport13 as transport
from gz.msgs10.double_pb2 import Double
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64

from competition_user.closed_loop.actuator import ThrustCommand
from competition_user.closed_loop.timing import CommandHistory
from competition_user.closed_loop.dynamics import VesselDynamics, VesselState, DisturbanceEstimator, wrap


# 所有值是牛顿；诊断移动约十米，不执行旧靠泊状态机。
PHASES = (("coast", 4., (0., 0.)), ("forward_40N", 6., (40., 40.)),
          ("passive_stop", 6., (0., 0.)), ("forward_80N", 6., (80., 80.)),
          ("active_brake", 6., None), ("positive_yaw", 4., (-50., 50.)),
          ("active_yaw_brake", 6., None), ("end_zero", 3., (0., 0.)))


class ActuatorProbe(Node):
    def __init__(self, output, control_period_s=.05, yaw_brake_s=6.):
        super().__init__("closed_loop_actuator_probe", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.output = Path(output)
        self.control_period_s = control_period_s
        self.phases = tuple((name, yaw_brake_s if name == "active_yaw_brake" else duration, force)
                            for name, duration, force in PHASES)
        config = Path(get_package_share_directory("competition_user")) / "config/vessel_dynamics.yaml"
        self.model = VesselDynamics.load(config)
        self.observer = DisturbanceEstimator(self.model)
        self.publishers_pair = [self.create_publisher(Float64, item.topic, 10)
                                for item in (self.model.actuator.left, self.model.actuator.right)]
        self.create_subscription(Odometry, "/team/localization/odometry", self.odometry, 1)
        self.state = None
        self.started_sim = None
        self.latest_stamp = None
        self.applied = ThrustCommand(0., 0.)
        self.command_history = CommandHistory()
        self.rows = []
        self.predictions = []
        self.pending = []
        self.last_prediction = -math.inf
        self.phase = "WAIT_NAVIGATION"
        self.phase_elapsed = 0.
        self.finished = False
        self.lock = Lock()
        self.feedback = defaultdict(list)
        self.gz = transport.Node()
        for side in ("left", "right"):
            if not self.gz.subscribe(Double, f"/wamv/thrusters/{side}/thrust/ang_vel",
                                     lambda msg, side=side: self.receive_feedback(msg, side)):
                raise RuntimeError("无法订阅推进器实际反馈：" + side)
        self.timer = self.create_timer(control_period_s, self.tick)
        self.started_wall = time.monotonic()

    def odometry(self, message):
        if message.header.frame_id != "team_enu":
            raise ValueError("执行诊断里程计不是 ENU")
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        q, p, velocity = message.pose.pose.orientation, message.pose.pose.position, message.twist.twist
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        state = VesselState(p.x, p.y, yaw, velocity.linear.x, velocity.linear.y, velocity.angular.z)
        self.observer.observe(state, stamp, self.command_history)
        self.state, self.latest_stamp = state, stamp
        remaining = []
        for target_stamp, predicted, initial_phase in self.pending:
            if stamp >= target_stamp:
                self.predictions.append({"phase": initial_phase, "stamp_sim_s": stamp,
                                         "position_error_m": math.hypot(state.x - predicted.x, state.y - predicted.y),
                                         "surge_error_mps": state.surge - predicted.surge,
                                         "sway_error_mps": state.sway - predicted.sway,
                                         "yaw_error_rad": wrap(state.yaw - predicted.yaw)})
            else:
                remaining.append((target_stamp, predicted, initial_phase))
        self.pending = remaining

    def receive_feedback(self, message, side):
        thruster = self.model.actuator.left if side == "left" else self.model.actuator.right
        force = self.applied.left_n if side == "left" else self.applied.right_n
        # 固定推力阶段跳过 ROS/Gazebo 命令边沿的传输间隔。
        if self.phase in ("forward_40N", "forward_80N", "positive_yaw") and self.phase_elapsed >= .5:
            with self.lock:
                self.feedback[side].append({"phase": self.phase, "force_n": force, "actual_feedback_rad_s": message.data,
                                           "expected_feedback_rad_s": thruster.feedback_from_force(force)})

    def publish(self, command):
        values = self.model.actuator.published_values(command)
        for publisher, value in zip(self.publishers_pair, values):
            publisher.publish(Float64(data=value))
        self.applied = command
        self.command_history.record(self.get_clock().now().nanoseconds/1e9, command)

    def tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if self.state is None:
            return
        if now - self.latest_stamp > .3:
            self.publish(ThrustCommand(0., 0.))
            raise RuntimeError("执行诊断定位失联")
        if self.started_sim is None:
            self.started_sim = now
        state = self.command_history.propagate(self.model, self.state, self.latest_stamp,
            max(now, self.latest_stamp), self.observer.estimate)
        elapsed, offset = now - self.started_sim, 0.
        phase = None
        for name, duration, force in self.phases:
            if elapsed < offset + duration:
                phase = name, elapsed - offset, force
                break
            offset += duration
        if phase is None:
            self.publish(ThrustCommand(0., 0.))
            self.finish()
            return
        name, phase_elapsed, force = phase
        if name != self.phase:
            self.get_logger().info(f"ACTUATOR_PROBE: {name}")
        self.phase, self.phase_elapsed = name, phase_elapsed
        command = self.model.actuator.command(*force) if force is not None else self.model.speed_command(
            state, 0., 0., self.observer.estimate)
        self.publish(command)
        self.rows.append({"phase": name, "phase_elapsed_s": phase_elapsed, "stamp_sim_s": now,
                          "state": state.__dict__, "measurement_stamp_sim_s": self.latest_stamp,
                          "measurement_age_s": now-self.latest_stamp, "published_force_n": command.__dict__,
                          "disturbance": self.observer.estimate.__dict__})
        if force is not None and name != "end_zero" and phase_elapsed < next(d for n, d, _ in self.phases if n == name) - 1.2:
            if now - self.last_prediction >= .5:
                predicted = self.model.predict(state, command, 1., disturbance=self.observer.estimate)[-1]
                self.pending.append((now + 1., predicted, name))
                self.last_prediction = now

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.timer.cancel()
        with self.lock:
            feedback = dict(self.feedback)
        errors = [abs(row["actual_feedback_rad_s"] - row["expected_feedback_rad_s"])
                  for rows in feedback.values() for row in rows]
        last_brake = [row for row in self.rows if row["phase"] == "active_brake" and row["phase_elapsed_s"] > 5.]
        last_turn = [row for row in self.rows if row["phase"] == "positive_yaw" and row["phase_elapsed_s"] > 2.]
        summary = {"probe_completed": True, "official_acceptance": False,
                   "control_period_s": self.control_period_s, "phases": self.phases,
                   "feedback_samples": {side: len(feedback.get(side, [])) for side in ("left", "right")},
                   "maximum_feedback_error_rad_s": max(errors) if errors else None,
                   "unit_contract_passed": bool(errors and all(feedback.get(side) for side in ("left", "right")) and max(errors) < 1e-5),
                   "braked_surge_abs_p95_mps": float(np.percentile([abs(row["state"]["surge"]) for row in last_brake], 95)) if last_brake else None,
                   "positive_yaw_rate_mean_rad_s": float(np.mean([row["state"]["yaw_rate"] for row in last_turn])) if last_turn else None,
                   "one_second_prediction_position_p95_m": float(np.percentile([row["position_error_m"] for row in self.predictions], 95)) if self.predictions else None,
                   "elapsed_wall_s": time.monotonic() - self.started_wall,
                   "note": "仅验证执行单位与短时动力学，不代表导航或官方成绩"}
        self.output.mkdir(parents=True, exist_ok=True)
        for filename, data in (("probe_samples.json", {"navigation_and_command": self.rows, "feedback": feedback,
                                                     "prediction_errors": self.predictions}), ("probe_result.json", summary)):
            temporary = (self.output / filename).with_suffix(".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.output / filename)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--control-period-s", type=float, default=.05)
    parser.add_argument("--yaw-brake-s", type=float, default=6.)
    args = parser.parse_args()
    rclpy.init()
    node = ActuatorProbe(args.output, args.control_period_s, args.yaw_brake_s)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(ThrustCommand(0., 0.))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
