"""现有感知接口适配到单一控制链；不导入旧状态机或另一套指令保护。"""

from collections import Counter, deque
from dataclasses import replace
import json
import math
from pathlib import Path
import signal
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
from phase16z_perception_msgs.msg import DynamicTrackArray
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64, String

from competition_user.runtime_safety import pose_at_stamp
from .timing import CommandHistory
from .sim_clock import SimulationClockReader
from .force_sequence import TimedForceSequence
from .force_executor import ApprovedForceExecutor
from .actuator import ThrustCommand
from .dynamics import VesselDynamics, VesselState, DisturbanceEstimator, wrap
from .geometry import Hull, CollisionField
from .mission import Mission, MissionState
from .navigation import ObservationMap, RoutePlanner
from .planner import ThrustPlanner, Guidance, MovingObstacle


BASE_FRAMES = {"wamv/wamv/base_link", "wamv/base_link"}
GRID_FRAMES = BASE_FRAMES | {"wamv/base_link_planar"}


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_from_quaternion(q):
    return math.atan2(2 * (q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ClosedLoopNode(Node):
    def __init__(self, simulation_clock=None, force_timer_node=None):
        super().__init__("closed_loop_berthing", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.sim_clock = simulation_clock if simulation_clock is not None else self.get_clock()
        self.declare_parameter("runtime_output_dir", "/tmp/closed_loop_control")
        self.output = Path(str(self.get_parameter("runtime_output_dir").value))
        self.output.mkdir(parents=True, exist_ok=True)
        self.model = VesselDynamics.load(Path(get_package_share_directory("competition_user")) / "config/vessel_dynamics.yaml")
        config = Path(get_package_share_directory("phase17_berthing_perception")) / "config"
        self.hull = Hull.load(config / "hull_collision_footprint_base_link.yaml")
        self.mission = Mission()
        self.observer = DisturbanceEstimator(self.model)
        self.map = ObservationMap()
        self.router = RoutePlanner(self.hull)
        self.planner = ThrustPlanner(self.model, self.hull)
        self.navigation_history = deque(maxlen=500)
        self.receipts = {}
        self.state = None
        self.applied = ThrustCommand(0., 0.)
        self.last_command_stamp_sim_s = None
        self.command_history = CommandHistory()
        self.measurement_state = None
        self.measurement_stamp = None
        self.grid_radius = math.sqrt(2)*.2/2
        self.track_cache = {}
        self.rejected = Counter()
        self.pending_grid = None
        self.started_wall = time.monotonic()
        self.started_sim = None
        self.max_plan_s = self.max_cycle_s = 0.
        self.last_status_wall = -math.inf
        self.last_mode = None
        self.last_plan = None
        self.last_input_snapshot_sim = -math.inf
        self.last_input_snapshot_tracks = set()
        self.input_status = "WAITING_FOR_INPUTS"
        self.completion_written = False
        self.completion_payload = None
        self.execution_log = (self.output / "execution_log.jsonl").open("a", encoding="utf-8")
        self.left = self.create_publisher(Float64, self.model.actuator.left.topic, 10)
        self.right = self.create_publisher(Float64, self.model.actuator.right.topic, 10)
        self.force_execution_log = (self.output/'force_execution_log.jsonl').open('a',encoding='utf-8')
        self.force_executor = ApprovedForceExecutor(lambda:self.now_sim(),self._publish_force)
        # 独立时钟执行器只切换已批准的力；业务输入和求解仍顺序执行。
        self.force_timer_node = force_timer_node if force_timer_node is not None else self
        self.force_timer = self.force_timer_node.create_timer(.005,self.force_executor.tick,clock=self.sim_clock)
        self.status_publisher = self.create_publisher(String, "/team/planning/status", 10)
        self.completion_publisher = self.create_publisher(String, "/team/planning/completion", 10)
        # 短输入回调保留最近数据；同一控制周期内快照、模型和实际命令一致。
        self.create_subscription(Odometry, "/team/localization/odometry", self.odometry, 1)
        self.create_subscription(OccupancyGrid, "/team/perception/local_occupancy", self.occupancy,
                                 QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(DynamicTrackArray, "/team/perception/dynamic_tracks", self.tracks, 2)
        self.create_subscription(PoseStamped, "/world/obstacles/dock", self.dock_prior, 2)
        self.create_subscription(String, "/team/perception/berth_candidates_world", self.bays, 2)
        self.timer = self.create_timer(.1, self.tick, clock=self.sim_clock)
        self.get_logger().info("闭环重建模块已启动：实际牛顿值、统一受力预测、唯一指令出口")

    def now_sim(self):
        return self.sim_clock.now().nanoseconds / 1e9

    def accept(self, name, stamp, maximum_age):
        if stamp <= self.receipts.get(name, (-math.inf, 0.))[0]:
            self.rejected[name + "_NONINCREASING"] += 1
            return False
        if not -.05 <= self.now_sim() - stamp <= maximum_age:
            self.rejected[name + "_MEASUREMENT_AGE"] += 1
            return False
        self.receipts[name] = (stamp, time.monotonic())
        return True

    def odometry(self, message):
        if message.header.frame_id != "team_enu" or message.child_frame_id not in BASE_FRAMES:
            self.rejected["ODOMETRY_FRAME"] += 1
            return
        p, q, v = message.pose.pose.position, message.pose.pose.orientation, message.twist.twist
        values = (p.x, p.y, yaw_from_quaternion(q), v.linear.x, v.linear.y, v.angular.z)
        if not all(math.isfinite(value) for value in values):
            self.rejected["ODOMETRY_VALUES"] += 1
            return
        stamp = stamp_seconds(message.header.stamp)
        if not self.accept("odom", stamp, .75):
            return
        self.measurement_state = self.state = VesselState(*values)
        self.measurement_stamp = stamp
        self.observer.observe(self.state, stamp, self.command_history)
        self.navigation_history.append((stamp, p.x, p.y, self.state.yaw))
        if self.pending_grid is not None:
            grid, self.pending_grid = self.pending_grid, None
            self.occupancy(grid)

    def occupancy(self, message):
        if message.header.frame_id not in GRID_FRAMES:
            self.rejected["GRID_FRAME"] += 1
            return
        info = message.info
        if (info.width <= 0 or info.height <= 0 or len(message.data) != info.width * info.height
                or not math.isfinite(info.resolution) or info.resolution <= 0):
            self.rejected["GRID_LAYOUT"] += 1
            return
        p, q = info.origin.position, info.origin.orientation
        origin = p.x, p.y, yaw_from_quaternion(q)
        if not all(math.isfinite(value) for value in origin):
            self.rejected["GRID_ORIGIN"] += 1
            return
        stamp = stamp_seconds(message.header.stamp)
        pose = pose_at_stamp(self.navigation_history, stamp)
        if pose is None:
            self.pending_grid = message
            return
        if not self.accept("grid", stamp, 1.5):
            return
        grid = np.asarray(message.data, dtype=np.int8).reshape(message.info.height, message.info.width)
        self.map.update(grid, origin, pose, message.info.resolution)
        self.grid_radius = math.sqrt(2)*message.info.resolution/2

    def tracks(self, message):
        if message.header.frame_id != "team_enu":
            self.rejected["TRACK_FRAME"] += 1
            return
        stamp = stamp_seconds(message.header.stamp)
        if not self.accept("dynamic", stamp, 2.):
            return
        for track in message.tracks:
            values = (track.pose.position.x, track.pose.position.y, track.velocity.linear.x, track.velocity.linear.y)
            if not all(math.isfinite(v) for v in values):
                self.rejected["TRACK_VALUES"] += 1
                continue
            covariance = float(track.velocity_covariance_xx), float(track.velocity_covariance_yy)
            if not all(math.isfinite(value) and value >= 0. for value in covariance):
                self.rejected["TRACK_COVARIANCE"] += 1
                continue
            variance = max(covariance)
            obstacle = MovingObstacle(*values, velocity_std_mps=math.sqrt(variance))
            self.track_cache[int(track.track_id)] = (stamp, obstacle)

    def dock_prior(self, message):
        p, q = message.pose.position, message.pose.orientation
        pose = p.x, p.y, yaw_from_quaternion(q)
        if all(math.isfinite(v) for v in pose):
            self.mission.dock_prior = pose
        else:
            self.rejected["DOCK_PRIOR_VALUES"] += 1

    def bays(self, message):
        try:
            self.mission.accept_bays(json.loads(message.data), self.now_sim())
        except (ValueError, KeyError, TypeError) as error:
            self.rejected["BAY_PROTOCOL"] += 1
            self.get_logger().error("BAY_PROTOCOL: " + str(error))

    def input_capability(self, now):
        wall = time.monotonic()
        for key, age in (("odom", .75), ("grid", 1.5), ("dynamic", 2.)):
            receipt = self.receipts.get(key)
            if receipt is None or not -.05 <= now - receipt[0] <= age or wall - receipt[1] > 3.:
                return key + "_STALE_OR_MISSING"
        return "READY"

    def moving_obstacles(self, now):
        self.track_cache = {key: item for key, item in self.track_cache.items() if 0. <= now-item[0] <= 2.}
        return [replace(item, x=item.x+item.vx*(now-stamp), y=item.y+item.vy*(now-stamp))
                for stamp, item in self.track_cache.values()]

    def _publish_force(self, command, planned_stamp, source):
        left, right = self.model.actuator.published_values(command)
        self.left.publish(Float64(data=left))
        self.right.publish(Float64(data=right))
        self.applied = command
        self.last_command_stamp_sim_s = self.now_sim()
        self.command_history.record(self.last_command_stamp_sim_s, command)
        self.force_execution_log.write(json.dumps(dict(stamp_sim_s=self.last_command_stamp_sim_s,
            planned_stamp_sim_s=planned_stamp,source=source,force_n=command.__dict__),ensure_ascii=False)+'\n')
        return self.last_command_stamp_sim_s

    def publish_command(self, command):
        self.force_executor.replace_constant(command)

    def destroy_node(self):
        self.force_executor.close()
        self.force_timer_node.destroy_timer(self.force_timer)
        self.force_execution_log.close()
        return super().destroy_node()

    def tick(self):
        started = time.perf_counter()
        try:
            self.control_tick()
        finally:
            # 包含状态输出与完成事件的实际周期，不只计算轨迹的核心函数。
            self.max_cycle_s = max(self.max_cycle_s,time.perf_counter()-started,self.force_executor.max_cycle_s)
            if self.completion_payload is not None and self.completion_payload["max_control_cycle_s"] < self.max_cycle_s:
                self.completion_payload["max_control_cycle_s"] = self.max_cycle_s
                atomic_json(self.output / "completion.json", self.completion_payload)

    def control_tick(self):
        started, now = time.perf_counter(), self.now_sim()
        if self.started_sim is None and now > 0.:
            self.started_sim = now
        self.input_status = self.input_capability(now)
        if self.state is None or self.input_status.startswith("odom_"):
            self.publish_command(ThrustCommand(0., 0.))
            self.write_status()
            return
        # 从原始测量推进到当前执行时刻，重复tick不会反复积分同一测量。
        self.state = self.command_history.propagate(self.model, self.measurement_state,
            self.measurement_stamp, max(now, self.measurement_stamp), self.observer.estimate)
        self.mission.update(self.state, now, can_complete=None)
        if time.monotonic()-self.started_wall >= 900. and self.mission.state != MissionState.COMPLETE:
            self.mission.state, self.mission.reason = MissionState.FAILED, "OFFICIAL_WALL_DEADLINE"
        points = self.map.nearby((self.state.x, self.state.y))
        field = CollisionField(points, self.hull, self.grid_radius)
        guidance = self.mission.guidance(self.state)
        if self.input_status != "READY" or guidance is None or self.mission.state in (MissionState.FAILED, MissionState.COMPLETE):
            # 有导航时仍预测制动和侧移，交给同一出口；不叠加第二套零速保护。
            guidance = Guidance((self.state.x, self.state.y), 0., 0.)
        elif guidance.use_route and self.mission.state in (MissionState.NAVIGATE, MissionState.APPROACH):
            self.router.plan((self.state.x, self.state.y), guidance.goal_xy, points, now)
            goal = self.router.lookahead((self.state.x, self.state.y), 3. if self.mission.bay else 4.)
            if goal is not None:
                heading = math.atan2(goal[1]-self.state.y, goal[0]-self.state.x)
                error = wrap(heading-self.state.yaw)
                speed = guidance.desired_surge * max(0., math.cos(error))**2
                guidance = replace(guidance, goal_xy=goal, desired_surge=speed,
                                   desired_yaw_rate=max(-.5, min(.5, .8*error)))
        moving = self.moving_obstacles(now)
        planning_state = self.state
        def execution_snapshot():
            stamp = self.now_sim()
            current = self.command_history.propagate(self.model,self.measurement_state,
                self.measurement_stamp,max(stamp,self.measurement_stamp),self.observer.estimate)
            advanced = [replace(item,x=item.x+item.vx*(stamp-now),y=item.y+item.vy*(stamp-now)) for item in moving]
            return current,advanced,stamp
        plan = self.planner.choose(self.state,guidance,field,moving,self.observer.estimate,
            execution_snapshot=execution_snapshot)
        self.last_plan = plan
        decision = self.planner.last_diagnostics
        if plan.status == 'SAFE' and 'selected_force_durations_s' in decision:
            self.force_executor.install(TimedForceSequence(decision['selected_controls_n'],
                decision['selected_force_durations_s'],decision['prediction_stamp_sim_s']))
        else:
            self.publish_command(plan.command)
        published_stamp, published_force = self.force_executor.snapshot()
        execution_stamp = published_stamp
        self.state = self.command_history.propagate(self.model,self.measurement_state,
            self.measurement_stamp,max(execution_stamp,self.measurement_stamp),self.observer.estimate)
        # 位置、船速和执行时间采用同一真实发布时间；独立执行线程的换力也已记入历史。
        bay_fresh = self.mission.bay is not None and execution_stamp-self.mission.bay_stamp <= 3.
        current_clearance = field.clearances([self.state.pose])[0]
        self.mission.check_completion(self.state,execution_stamp,
            can_complete=self.input_capability(execution_stamp)=="READY" and bay_fresh
                         and plan.status=="SAFE" and current_clearance>=0.)
        # 原输入保存放在指令发布后，I/O仍计入整个周期，不延迟已验安全的动作。
        snapshot = self.save_decision_inputs(now,plan.status,field,
            terminal=guidance.feedback == "terminal",slow=plan.duration_s>.25)
        self.max_plan_s = max(self.max_plan_s, time.perf_counter()-started)
        self.execution_log.write(json.dumps({"stamp_sim_s": now, "measurement_stamp_sim_s": self.measurement_stamp,
            "measurement_age_s": now-self.measurement_stamp, "measurement_vessel": self.measurement_state.__dict__, "state": self.mission.state.value,
            "input_status": self.input_status, "vessel": planning_state.__dict__, "guidance": guidance.__dict__,
            "execution_stamp_sim_s": execution_stamp, "execution_vessel": self.state.__dict__,
            "prediction_stamp_sim_s": self.planner.last_diagnostics.get('prediction_stamp_sim_s',now),
            "prediction_initial_vessel": plan.trajectory[0].__dict__,
            "mission_reason": self.mission.reason,
            "bay_pending_occupancy": self.mission.pending_occupancy,
            "bay_occupied_measurements": self.mission.occupied_measurements,
            "selected_bay": self.mission.bay.__dict__ if self.mission.bay is not None else None,
            "bay_measurement_stamp_sim_s": self.mission.bay_stamp if self.mission.bay is not None else None,
            "geometry_stamp_sim_s": self.mission.geometry_stamp if math.isfinite(self.mission.geometry_stamp) else None,
            "published_stamp_sim_s": published_stamp,
            "plan_status": plan.status, "setpoint": plan.setpoint, "published_force_n": published_force.__dict__,
            "prediction_trajectory": [state.__dict__ for state in plan.trajectory],
            "decision_diagnostics": self.planner.last_diagnostics,
            "static_clearance_m": plan.minimum_static_clearance_m, "dynamic_clearance_m": plan.minimum_dynamic_clearance_m,
            "safe_candidate_count": plan.safe_candidate_count, "disturbance": self.observer.estimate.__dict__,
            "moving_obstacles": [dict(track_id=key, **item.__dict__) for key, item in zip(self.track_cache, moving)],
            "static_input_snapshot": snapshot, "static_point_radius_m": self.grid_radius,
            "plan_duration_s": time.perf_counter()-started}, ensure_ascii=False)+"\n")
        if self.mission.state == MissionState.COMPLETE and not self.completion_written:
            self.complete()
        self.write_status()

    def save_decision_inputs(self, now, status, field, terminal=False, slow=False):
        tracks = set(self.track_cache)
        needed = terminal or slow or status != "SAFE" or tracks != self.last_input_snapshot_tracks
        if not needed or (not slow and now-self.last_input_snapshot_sim < 1.):
            return None
        directory = self.output/"decision_inputs"
        directory.mkdir(exist_ok=True)
        path = directory/f"inputs_{round(now*1e9)}.npz"
        # 保存本周期真正参与规划的占用点，便于失败后精确重放；不反馈任何真值。
        np.savez(path, points=field.points, point_radius_m=field.point_radius)
        self.last_input_snapshot_sim, self.last_input_snapshot_tracks = now, tracks
        return str(path.relative_to(self.output))

    def complete(self):
        payload = {"task_completed": True, "state": self.mission.state.value, "bay_id": self.mission.bay.bay_id,
                   "stamp_sim_s": self.now_sim(),
                   "position_xy_m": [self.state.x, self.state.y], "speed_mps": self.state.speed,
                   "elapsed_wall_s": time.monotonic()-self.started_wall,
                   "elapsed_sim_s": self.now_sim()-self.started_sim,
                   "max_single_plan_duration_s": self.max_plan_s, "max_control_cycle_s": self.max_cycle_s,
                   "max_force_execution_s": self.force_executor.max_cycle_s,
                   "max_force_schedule_lateness_sim_s": self.force_executor.max_lateness_s,
                   "force_schedule_transition_count": self.force_executor.transitions,
                   "source": "production_navigation_and_verified_berth"}
        atomic_json(self.output / "completion.json", payload)
        self.completion_payload = payload
        self.completion_written = True
        self.completion_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.execution_log.flush()
        with self.force_executor.lock:
            self.force_execution_log.flush()
        self.get_logger().info("TASK_COMPLETED: " + json.dumps(payload, ensure_ascii=False))

    def write_status(self):
        wall = time.monotonic()
        if wall-self.last_status_wall < .5:
            return
        self.last_status_wall = wall
        bay = None if self.mission.bay is None else self.mission.bay.__dict__
        pose = None if self.state is None else [self.state.x, self.state.y]
        payload = {"controller_version": "closed_loop_rebuild", "state": self.mission.state.value,
                   "reason": self.mission.reason, "input_status": self.input_status,
                   "position_xy_m": pose, "measurement_stamp_sim_s": self.measurement_stamp,
                   "measurement_age_s": None if self.measurement_stamp is None else self.now_sim()-self.measurement_stamp,
                   "speed_mps": None if self.state is None else self.state.speed,
                   "elapsed_wall_s": wall-self.started_wall,
                   "elapsed_sim_s": None if self.started_sim is None else self.now_sim()-self.started_sim,
                   "selected_bay": bay, "navigation_completed": self.completion_written,
                   "max_single_plan_duration_s": self.max_plan_s, "max_control_cycle_s": self.max_cycle_s,
                   "plan_status": None if self.last_plan is None else self.last_plan.status,
                   "published_force_n": self.applied.__dict__, "input_rejections": dict(self.rejected),
                   "static_map_cells": len(self.map.occupied), "dynamic_track_count": len(self.track_cache)}
        payload["route_status"] = self.router.status
        payload['max_force_execution_s'] = self.force_executor.max_cycle_s
        payload['max_force_schedule_lateness_sim_s'] = self.force_executor.max_lateness_s
        payload['force_schedule_transition_count'] = self.force_executor.transitions
        atomic_json(self.output / "runtime_status.json", payload)
        self.status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.execution_log.flush()
        with self.force_executor.lock:
            self.force_execution_log.flush()
        mode = (payload["state"], payload["input_status"], payload["plan_status"])
        if mode != self.last_mode:
            self.get_logger().info("CONTROL: " + json.dumps(payload, ensure_ascii=False))
            self.last_mode = mode


def main(args=None):
    rclpy.init(args=args)
    clock_reader = SimulationClockReader()
    node = ClosedLoopNode(simulation_clock=clock_reader.clock,force_timer_node=clock_reader.node)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.2)
    finally:
        node.publish_command(ThrustCommand(0., 0.))
        node.execution_log.close()
        node.destroy_node()
        clock_reader.close()
        if rclpy.ok():
            rclpy.shutdown()
