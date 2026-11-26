"""独立读取仿真评估真值及接触，绝不向生产节点传递真值或发布推力。"""

from collections import Counter, deque
import base64
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import signal
import sys
from threading import RLock
import time

import gz.transport13 as transport
import numpy as np
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.pose_v_pb2 import Pose_V
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointCloud2, Imu
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from phase16z_perception_msgs.msg import DynamicDetectionArray, DynamicTrackArray
from official_evaluation import completion_snapshot


def simulation_stamp(message):
    return message.header.stamp.sec + message.header.stamp.nsec * 1e-9


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class OfficialMonitor(Node):
    def __init__(self, output, sensor_trace=False):
        super().__init__("closed_loop_independent_evaluation")
        self.out = Path(output)
        self.inputs_path = self.out / "input_snapshots"
        self.inputs_path.mkdir(exist_ok=True)
        self.manifest = json.loads((self.out / "scene/manifest.json").read_text(encoding="utf-8"))
        self.lock = RLock()
        self.sensor_stream = (self.out / "sensor_motion_history.jsonl").open("w", encoding="utf-8") if sensor_trace else None
        self.started = time.monotonic()
        self.truth = deque(maxlen=300)
        self.completion_truth = None
        self.completion_pending_stamp = None
        self.current_pose = None
        self.speed = None
        self.truth_stamp = None
        self.counts = Counter()
        self.events = []
        self.last_contact = {}
        self.clouds = {}
        self.status = {}
        self.pose_names = []
        self.navigation = {}
        self.boat_truth = {}
        self.last_tracks = {}
        self.last_grid_sample_stamp = -math.inf
        self.gz = transport.Node()
        world = self.manifest["world_name"]
        self.pose_subscription_ok = self.gz.subscribe(Pose_V, f"/world/{world}/dynamic_pose/info", self.pose_callback)
        # 原始船模的 contact_sensor 明确发布到 /vrx/contacts。
        self.contact_subscription_ok = self.gz.subscribe(Contacts, "/vrx/contacts", self.contacts_callback)
        for name in ("back", "forward", "front", "left", "right"):
            topic = f"/wamv/sensors/lidars/{name}_lidar_sensor/points"
            self.create_subscription(PointCloud2, topic, lambda msg, key=name: self.cloud_callback(msg, key), 10)
        for name, topic in (("candidate", "/team/perception/status"),
                            ("localization", "/team/localization/status"),
                            ("tracking", "/team/perception/dynamic_tracking_status"),
                            ("berthing", "/team/perception/berthing_status"),
                            ("environment", "/team/perception/local_environment_status")):
            self.create_subscription(String, topic, lambda msg, key=name: self.status_callback(msg, key), 10)
        self.create_subscription(DynamicTrackArray, "/team/perception/dynamic_tracks", self.tracks_callback, 10)
        self.create_subscription(OccupancyGrid, "/team/perception/local_occupancy", self.grid_callback, 10)
        self.create_subscription(Imu, "/wamv/sensors/imu/imu/data", self.imu_callback, 10)
        self.create_subscription(Odometry, "/team/localization/odometry", self.odom_callback, 10)
        self.create_subscription(String, "/team/planning/completion", self.completion_callback, 10)
        self.create_subscription(String, "/team/perception/berth_candidates_world", self.berth_protocol_callback, 10)
        if sensor_trace:
            self.create_subscription(NavSatFix, "/wamv/sensors/gps/gps/fix", self.gps_trace, 20)
            self.create_subscription(TwistStamped, "/wamv/sensors/gps/gps/velocity", self.velocity_trace, 20)
            self.create_subscription(DynamicDetectionArray, "/team/perception/dynamic_detections_base", self.detections_trace, 10)
        self.create_timer(1., self.write)

    def pose_callback(self, message):
        with self.lock:
            self.counts["truth_pose_messages"] += 1
            if not self.pose_names:
                self.pose_names = [pose.name for pose in message.pose][:80]
            stamp = simulation_stamp(message)
            self.boat_truth = {pose.name: [pose.position.x, pose.position.y]
                               for pose in message.pose if re.fullmatch(r"boat\d+", pose.name)}
            pose = next((item for item in message.pose if item.name == "wamv"), None)
            if pose is None:
                return
            stamp = simulation_stamp(message)
            if self.truth and stamp <= self.truth[-1][0]:
                return
            x, y = pose.position.x, pose.position.y
            self.truth.append((stamp, x, y))
            recent = [row for row in self.truth if stamp - row[0] <= .20]
            if len(recent) >= 2 and stamp > recent[0][0]:
                vx = (x - recent[0][1]) / (stamp - recent[0][0])
                vy = (y - recent[0][2]) / (stamp - recent[0][0])
                self.speed = math.hypot(vx, vy)
            q = pose.orientation
            yaw = math.atan2(2. * (q.w * q.z + q.x * q.y), 1. - 2. * (q.y * q.y + q.z * q.z))
            self.current_pose = [x, y, yaw]
            self.truth_stamp = stamp
            self.trace_sensor("independent_truth", stamp, {"xyz": [x, y, pose.position.z], "xyzw": [q.x, q.y, q.z, q.w]})
            self.capture_completion_truth()

    def completion_callback(self, message):
        report = json.loads(message.data)
        if report.get("task_completed") is True:
            with self.lock:
                self.completion_pending_stamp = float(report["stamp_sim_s"])
                self.capture_completion_truth()

    def capture_completion_truth(self):
        stamp = self.completion_pending_stamp
        if stamp is None or self.completion_truth is not None:
            return
        self.completion_truth = completion_snapshot(self.truth, stamp)

    def contacts_callback(self, message):
        with self.lock:
            self.counts["contact_stream_messages"] += 1
            stamp = simulation_stamp(message)
            for contact in message.contact:
                first, second = contact.collision1.name, contact.collision2.name
                own_first, own_second = "wamv" in first, "wamv" in second
                if own_first == own_second:
                    continue
                other = second if own_first else first
                if "water_collision_plane" in other:
                    continue
                root = other.split("::")[0]
                category = "berth" if "dock" in other.lower() else "dynamic" if re.search(r"boat\d", other) else "static"
                previous = self.last_contact.get(root)
                if previous is not None and stamp - previous[0] <= 1.:
                    event = self.events[previous[1]]
                    event["last_stamp_sim_s"] = stamp
                    event["messages"] += 1
                else:
                    age = None if self.truth_stamp is None else abs(stamp - self.truth_stamp)
                    event = {"other": root, "collision_name": other, "category": category,
                             "first_stamp_sim_s": stamp, "last_stamp_sim_s": stamp,
                             "first_speed_mps": self.speed if age is not None and age <= .25 else None,
                             "truth_speed_age_s": age, "first_position_xy_m": self.current_pose,
                             "messages": 1}
                    self.events.append(event)
                self.last_contact[root] = (stamp, self.events.index(event))

    def cloud_callback(self, message, name):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        row = self.clouds.setdefault(name, {"count": 0, "first_stamp_sim_s": stamp})
        row.update(last_stamp_sim_s=stamp, width=message.width, height=message.height, frame_id=message.header.frame_id)
        row["count"] += 1
        if stamp - row.get("last_finite_sample_sim_s", -math.inf) >= 5.:
            # 定期读取真实点云有效率，不把模型文件中的规格等同于收到有效回波。
            fields = {field.name: field for field in message.fields}
            if all(key in fields and fields[key].datatype == 7 for key in ("x", "y", "z")):
                dtype = np.dtype({"names": ["x", "y", "z"],
                                  "formats": [">f4" if message.is_bigendian else "<f4"] * 3,
                                  "offsets": [fields[key].offset for key in ("x", "y", "z")],
                                  "itemsize": message.point_step})
                points = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                                    strides=(message.row_step, message.point_step))
                finite = np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"])
                row.update(finite_points_sample=int(finite.sum()), total_points_sample=int(finite.size),
                           last_finite_sample_sim_s=stamp)
                slot = int(stamp // 5.) % 2
                # 只保留最后两组原始快照，用于失败后定位，避免重复跑整程取数据。
                xyz = np.stack([points[key] for key in ("x", "y", "z")], axis=2)
                np.savez_compressed(self.inputs_path / f"cloud_{name}_{slot}.npz", xyz=xyz, stamp=stamp,
                                    frame=message.header.frame_id, navigation=json.dumps(self.navigation))

    def grid_callback(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if stamp - self.last_grid_sample_stamp < 5.:
            return
        self.last_grid_sample_stamp = stamp
        q = message.info.origin.orientation
        yaw = math.atan2(2. * (q.w * q.z + q.x * q.y), 1. - 2. * (q.y * q.y + q.z * q.z))
        np.savez_compressed(self.inputs_path / "grid_latest.npz", stamp=stamp,
                            grid=np.asarray(message.data, dtype=np.int8).reshape(message.info.height, message.info.width),
                            resolution=message.info.resolution,
                            origin=[message.info.origin.position.x, message.info.origin.position.y, yaw],
                            navigation=json.dumps(self.navigation))

    def imu_callback(self, message):
        q = message.orientation
        self.navigation["imu_xyzw"] = [q.x, q.y, q.z, q.w]
        self.trace_sensor("imu", message.header.stamp.sec+message.header.stamp.nanosec/1e9,
                          {"xyzw": [q.x, q.y, q.z, q.w], "angular": [message.angular_velocity.x,
                           message.angular_velocity.y, message.angular_velocity.z]})

    def odom_callback(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        self.navigation["odom_xyz_xyzw"] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        v = message.twist.twist
        self.trace_sensor("odometry", message.header.stamp.sec+message.header.stamp.nanosec/1e9,
                          {"xyz": [p.x, p.y, p.z], "xyzw": [q.x, q.y, q.z, q.w],
                           "linear": [v.linear.x, v.linear.y, v.linear.z],
                           "angular": [v.angular.x, v.angular.y, v.angular.z]})

    def trace_sensor(self, stream, stamp, payload):
        if self.sensor_stream is not None:
            # 独立高频诊断只写证据，绝不向生产反馈世界真值。
            with self.lock:
                self.sensor_stream.write(json.dumps({"stream": stream, "stamp": stamp, **payload}, ensure_ascii=False)+"\n")

    def gps_trace(self, message):
        self.trace_sensor("gps_fix", message.header.stamp.sec+message.header.stamp.nanosec/1e9,
                          {"lla": [message.latitude, message.longitude, message.altitude]})

    def velocity_trace(self, message):
        v = message.twist.linear
        self.trace_sensor("gps_velocity", message.header.stamp.sec+message.header.stamp.nanosec/1e9,
                          {"enu": [v.x, v.y, v.z]})

    def detections_trace(self, message):
        # 完整CDR保留每个候选的原时间和几何特征，避免只存公共轨迹而无法解释确认延迟。
        self.trace_sensor("base_detections", message.header.stamp.sec+message.header.stamp.nanosec/1e9,
                          {"cdr_b64": base64.b64encode(serialize_message(message)).decode("ascii")})

    def status_callback(self, message, name):
        try:
            self.status[name] = json.loads(message.data)
            if name == "berthing":
                self.record_berthing_message("berthing_status", self.status[name])
        except json.JSONDecodeError:
            self.counts["bad_status_json"] += 1

    def record_berthing_message(self, kind, payload):
        # 独立保存实际收到的协议和墙钟，不把最新摘要当成完整历史。
        with (self.out / "berthing_protocol_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"elapsed_wall_s": time.monotonic()-self.started,
                                     "kind": kind, "payload": payload}, ensure_ascii=False)+"\n")

    def berth_protocol_callback(self, message):
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self.counts["bad_berth_protocol_json"] += 1
            return
        self.record_berthing_message("berth_candidates_world", payload)

    def tracks_callback(self, message):
        self.counts["tracks_messages"] += 1
        if message.tracks:
            self.counts["nonempty_tracks_messages"] += 1
        self.last_tracks = {"stamp_sim_s": message.header.stamp.sec+message.header.stamp.nanosec/1e9,
            "tracks": [{"track_id": int(track.track_id), "xy": [track.pose.position.x, track.pose.position.y],
                        "velocity_xy": [track.velocity.linear.x, track.velocity.linear.y],
                        "velocity_covariance_xy": [track.velocity_covariance_xx, track.velocity_covariance_yy],
                        "confidence": float(track.confidence), "missed_count": int(track.missed_count)}
                       for track in message.tracks]}

    def write(self):
        with self.lock:
            payload = {"elapsed_wall_s": time.monotonic() - self.started, "truth_stamp_sim_s": self.truth_stamp,
                       "truth_position_xyyaw": self.current_pose, "truth_speed_mps": self.speed,
                       "pose_names_sample": self.pose_names, "pose_subscription_ok": self.pose_subscription_ok,
                       "contact_subscription_ok": self.contact_subscription_ok, "counts": dict(self.counts),
                       "events": deepcopy(self.events), "lidar": deepcopy(self.clouds), "perception": deepcopy(self.status),
                       "navigation": deepcopy(self.navigation), "completion_truth": deepcopy(self.completion_truth)}
        atomic_json(self.out / "monitor.json", payload)
        # 真值与公共轨迹仅写独立证据文件，不发布回生产；失败后可区分漏检和误报。
        with self.lock:
            trace = {"stamp_sim_s": self.truth_stamp, "own_xyyaw": self.current_pose, "speed_mps": self.speed,
                     "boat_truth": deepcopy(self.boat_truth), "public_tracks": deepcopy(self.last_tracks),
                     "navigation": deepcopy(self.navigation)}
        with (self.out / "world_state_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trace, ensure_ascii=False)+"\n")


def main():
    rclpy.init()
    node = OfficialMonitor(sys.argv[1], sensor_trace="--sensor-trace" in sys.argv[2:])
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.2)
    finally:
        node.write()
        if node.sensor_stream is not None:
            node.sensor_stream.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
