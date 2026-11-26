"""五路点云码头精定位、多泊位状态 ROS2 节点。"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import threading
import time

from ament_index_python.packages import get_package_share_directory
from phase16z_s3_runtime.s3_ownship_geometry import load_ownship_geometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from .berthing_core import (
    LIDAR_TOPICS,
    aggregate_xy,
    base_pose_to_world,
    covariance_xyyaw,
    infer_bay_occupancy,
    leveling_rotation,
    load_extrinsics,
    make_berth_candidates,
    points_to_base,
    pose_confidence,
    quaternion_to_rpy,
    world_pose_to_base,
    wrap_angle,
    yaw_to_quaternion,
)
from .dock_localizer import (
    FrameScorer,
    load_template,
    refine_hypothesis_with_coarse_prior,
    register_with_coarse_prior,
    refine_icp_with_prior,
)
from .input_sync import select_fresh_inputs
from .navigation_history import TimestampedPoseBuffer, pose_from_odometry, base_pose, reproject_points, snapshot_poses


GPS_TOPIC = "/wamv/sensors/gps/gps/fix"
IMU_TOPIC = "/wamv/sensors/imu/imu/data"
DOCK_PRIOR_TOPIC = "/world/obstacles/dock"
LOCAL_BERTH_TOPIC = "/team/perception/berth_candidates"
WORLD_BERTH_TOPIC = "/team/perception/berth_candidates_world"
DOCK_POSE_TOPIC = "/team/perception/dock_pose"
STATUS_TOPIC = "/team/perception/berthing_status"


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class BerthingPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("phase17_berthing_perception")
        package_share = Path(get_package_share_directory("phase17_berthing_perception"))
        self.declare_parameter("extrinsics_path", str(package_share / "config" / "static_extrinsics.yaml"))
        self.declare_parameter("self_geometry_path", str(package_share / "config" / "visual_self_geometry.yaml"))
        self.declare_parameter("dock_template_path", str(package_share / "config" / "full_dock_collision_points.csv"))
        self.declare_parameter("receipt_path", "")
        self.declare_parameter("dock_yaw_override_rad", -1.658)
        self.declare_parameter("use_topic_dock_orientation", True)
        self.declare_parameter("sync_tolerance_ms", 100.0)
        self.declare_parameter("minimum_lidar_inputs", 3)
        self.declare_parameter("maximum_lidar_age_ms", 350.0)
        self.declare_parameter("navigation_tolerance_ms", 150.0)
        self.declare_parameter("dock_prior_max_age_s", 3.0)
        self.declare_parameter("maximum_dock_range_m", 35.0)
        self.declare_parameter("geometry_score_threshold", 0.18)
        self.declare_parameter("minimum_geometry_support", 0.65)
        self.declare_parameter("minimum_active_groups", 2)
        self.declare_parameter("occupancy_minimum_cells", 5)
        self.declare_parameter("entrance_offset_m", -8.0)
        self.declare_parameter("output_stale_s", 0.5)
        self.declare_parameter("maximum_processing_s", 0.40)
        self.declare_parameter("base_frame", "wamv/base_link")
        self.declare_parameter("world_frame", "team_enu")

        self.extrinsics = load_extrinsics(str(self.get_parameter("extrinsics_path").value))
        self.self_geometry = load_ownship_geometry(str(self.get_parameter("self_geometry_path").value))
        self.template, self.template_groups = load_template(
            Path(str(self.get_parameter("dock_template_path").value))
        )
        receipt_text = str(self.get_parameter("receipt_path").value).strip()
        self.receipt_path = Path(receipt_text) if receipt_text else None
        self.dock_yaw_override = float(self.get_parameter("dock_yaw_override_rad").value)
        self.use_topic_dock_orientation = bool(self.get_parameter("use_topic_dock_orientation").value)
        self.sync_tolerance_ns = int(float(self.get_parameter("sync_tolerance_ms").value) * 1e6)
        self.minimum_lidar_inputs = int(self.get_parameter("minimum_lidar_inputs").value)
        self.maximum_lidar_age_ns = int(
            float(self.get_parameter("maximum_lidar_age_ms").value) * 1e6
        )
        if not 1 <= self.minimum_lidar_inputs <= len(LIDAR_TOPICS) or self.maximum_lidar_age_ns <= 0:
            raise RuntimeError("LIDAR_FRESHNESS_PARAMETER_INVALID")
        self.navigation_tolerance_ns = int(float(self.get_parameter("navigation_tolerance_ms").value) * 1e6)
        self.dock_prior_max_age_ns = int(float(self.get_parameter("dock_prior_max_age_s").value) * 1e9)
        self.maximum_dock_range_m = float(self.get_parameter("maximum_dock_range_m").value)
        self.geometry_score_threshold = float(self.get_parameter("geometry_score_threshold").value)
        self.minimum_geometry_support = float(self.get_parameter("minimum_geometry_support").value)
        self.minimum_active_groups = int(self.get_parameter("minimum_active_groups").value)
        self.occupancy_minimum_cells = int(self.get_parameter("occupancy_minimum_cells").value)
        self.entrance_offset_m = float(self.get_parameter("entrance_offset_m").value)
        self.output_stale_s = float(self.get_parameter("output_stale_s").value)
        self.maximum_processing_s = float(self.get_parameter("maximum_processing_s").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.world_frame = str(self.get_parameter("world_frame").value)

        # 舍弃 DDS 中已被新帧替代的历史点云，避免在同步之前耗尽计算预算。
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        output_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.local_publisher = self.create_publisher(String, LOCAL_BERTH_TOPIC, output_qos)
        self.world_publisher = self.create_publisher(String, WORLD_BERTH_TOPIC, output_qos)
        self.pose_publisher = self.create_publisher(PoseWithCovarianceStamped, DOCK_POSE_TOPIC, output_qos)
        self.status_publisher = self.create_publisher(String, STATUS_TOPIC, output_qos)
        self.input_subscriptions = [
            self.create_subscription(Odometry, "/team/localization/odometry", self._navigation_callback, output_qos),
            self.create_subscription(PoseStamped, DOCK_PRIOR_TOPIC, self._dock_callback, qos),
        ]
        self.input_subscriptions.extend(
            self.create_subscription(
                PointCloud2,
                topic,
                lambda message, topic_name=topic: self._cloud_callback(message, topic_name),
                qos,
            )
            for topic in LIDAR_TOPICS
        )
        self.status_timer = self.create_timer(0.5, self._publish_status)

        self.lock = threading.RLock()
        self.latest_clouds: dict[str, PointCloud2] = {}
        self.navigation_buffer = TimestampedPoseBuffer()
        self.latest_dock: PoseStamped | None = None
        self.latest_dock_received_monotonic: float | None = None
        self.last_enqueued_stamp_ns = 0
        self.last_output_stamp_ns = 0
        self.last_output_wall = 0.0
        self.last_reason = "WAITING_FOR_INPUTS"
        self.last_input_issue = ""
        self.last_result: dict | None = None
        self.counts: dict[str, int] = defaultdict(int)
        self.queue: Queue[dict | None] = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, name="dock-latest-frame", daemon=True)
        self.worker.start()
        self.started_wall = time.time()
        self.get_logger().info("靠泊感知已启动：等待五路点云、共享导航和 dock 粗先验")

    def _navigation_callback(self, message: Odometry) -> None:
        try:
            pose = pose_from_odometry(message)
        except ValueError:
            with self.lock:
                self.counts["NAVIGATION_POSE_REJECTED"] += 1
            return
        with self.lock:
            self.navigation_buffer.add(pose)
            self.navigation_buffer.discard_before(pose.stamp_ns-2_000_000_000)
            self.counts["/team/localization/odometry"] += 1
            self._enqueue_if_complete_locked()

    def _dock_callback(self, message: PoseStamped) -> None:
        with self.lock:
            self.latest_dock = message
            self.latest_dock_received_monotonic = time.monotonic()
            self.counts[DOCK_PRIOR_TOPIC] += 1

    def _cloud_callback(self, message: PointCloud2, topic: str) -> None:
        with self.lock:
            self.latest_clouds[topic] = message
            self.counts[topic] += 1
            self._enqueue_if_complete_locked()

    def _enqueue_if_complete_locked(self) -> None:
        if not len(self.navigation_buffer) or self.latest_dock is None:
            return
        selection = select_fresh_inputs(
            {
                topic: stamp_ns(message.header.stamp)
                for topic, message in self.latest_clouds.items()
            },
            minimum_inputs=self.minimum_lidar_inputs,
            maximum_age_ns=self.maximum_lidar_age_ns,
        )
        if selection is None:
            self.last_input_issue = "INSUFFICIENT_FRESH_LIDARS"
            self.counts[self.last_input_issue] += 1
            return
        selected_topics, anchor, span_ns = selection
        if anchor <= self.last_enqueued_stamp_ns:
            return
        full_sync = (
            len(selected_topics) == len(LIDAR_TOPICS)
            and span_ns <= self.sync_tolerance_ns
        )
        if not full_sync:
            self.counts["PARTIAL_OR_ASYNC_LIDAR_FRAME"] += 1
        clouds = tuple((topic, self.latest_clouds[topic]) for topic in LIDAR_TOPICS if topic in selected_topics)
        try:
            poses = snapshot_poses(self.navigation_buffer, clouds, anchor)
        except ValueError:
            self.last_input_issue = "WAIT_NAVIGATION_INTERPOLATION_ENDPOINT"
            self.counts[self.last_input_issue] += 1
            return
        dock_stamp = stamp_ns(self.latest_dock.header.stamp)
        dock_delta = abs(anchor - dock_stamp) if dock_stamp > 0 else 0
        if dock_delta > self.dock_prior_max_age_ns:
            # 仿真桥接节点可能给静态 dock 粗先验盖系统时间，而传感器使用仿真时间。
            # 时间域明显不一致时改用本节点的到达时间，仍保留失联超时保护。
            received_age_ns = (
                int((time.monotonic() - self.latest_dock_received_monotonic) * 1e9)
                if self.latest_dock_received_monotonic is not None
                else self.dock_prior_max_age_ns + 1
            )
            if received_age_ns > self.dock_prior_max_age_ns:
                self.last_input_issue = "DOCK_PRIOR_STALE"
                self.counts[self.last_input_issue] += 1
                return
            self.counts["DOCK_STAMP_DOMAIN_MISMATCH"] += 1
        snapshot = {
            "anchor": anchor,
            "clouds": clouds,
            "poses": poses,
            "dock": self.latest_dock,
            "sync_span_ms": span_ns / 1e6,
            "input_mode": "FULL_SYNC" if full_sync else "PARTIAL_FRESH",
            "navigation_source": "/team/localization/odometry",
            "dock_delta_ms": dock_delta / 1e6,
        }
        try:
            self.queue.put_nowait(snapshot)
        except Full:
            try:
                self.queue.get_nowait()
            except Empty:
                pass
            self.queue.put_nowait(snapshot)
            self.counts["LATEST_FRAME_REPLACED"] += 1
        self.last_enqueued_stamp_ns = anchor

    @staticmethod
    def _read_xyz(message: PointCloud2) -> np.ndarray:
        values = point_cloud2.read_points_numpy(
            message, field_names=["x", "y", "z"], skip_nans=False
        )
        if getattr(values.dtype, "names", None):
            return np.column_stack((values["x"], values["y"], values["z"])).astype(np.float64)
        return np.asarray(values, dtype=np.float64).reshape(-1, 3)

    def _process_snapshot(self, snapshot: dict) -> None:
        started = time.monotonic()
        if self.get_clock().now().nanoseconds - int(snapshot["anchor"]) > int(self.output_stale_s * 1e9):
            with self.lock:
                self.counts["OLD_MEASUREMENT_SKIPPED"] += 1
            return
        deadline = time.perf_counter() + self.maximum_processing_s
        anchor_pose = snapshot["poses"][snapshot["anchor"]]
        vehicle = base_pose(anchor_pose)

        parts = []
        origins = []
        ownship_parts = []
        raw_points = 0
        for topic, message in snapshot["clouds"]:
            sensor_points = self._read_xyz(message)
            raw_points += len(sensor_points)
            part = points_to_base(sensor_points, self.extrinsics[topic])
            # 自身点在各自测量时刻过滤，然后再投影到共同锚点。
            ownship_parts.append(self.self_geometry.contains(part))
            source_pose = snapshot["poses"][stamp_ns(message.header.stamp)]
            parts.append(reproject_points(part, source_pose, anchor_pose))
            source_origin = reproject_points(self.extrinsics[topic][None, :3, 3], source_pose, anchor_pose)[0]
            origins.append(np.broadcast_to(source_origin, part.shape))
        points_base = np.concatenate(parts, axis=0) if parts else np.empty((0, 3), dtype=np.float64)
        points_level = points_base @ leveling_rotation(vehicle.roll_rad, vehicle.pitch_rad).T
        ownship_mask = np.concatenate(ownship_parts) if ownship_parts else np.zeros(len(points_base), dtype=bool)
        ray_origins = (np.concatenate(origins, axis=0) if origins else np.empty((0, 3))) @ leveling_rotation(vehicle.roll_rad, vehicle.pitch_rad).T

        distances = np.linalg.norm(points_level[:, :2], axis=1)
        localizer_mask = (
            np.isfinite(points_level).all(axis=1)
            & ~ownship_mask
            & (distances >= 1.5)
            & (distances <= self.maximum_dock_range_m)
            & (points_level[:, 2] >= -0.5)
            & (points_level[:, 2] <= 3.0)
        )
        features_xy = aggregate_xy(points_level[localizer_mask])

        dock = snapshot["dock"]
        dock_q = dock.pose.orientation
        _, _, message_dock_yaw = quaternion_to_rpy(dock_q.x, dock_q.y, dock_q.z, dock_q.w)
        dock_world_yaw = (
            message_dock_yaw if self.use_topic_dock_orientation else self.dock_yaw_override
        )
        prior = world_pose_to_base(
            dock.pose.position.x,
            dock.pose.position.y,
            dock_world_yaw,
            vehicle,
        )
        prior_range = math.hypot(prior["x"], prior["y"])

        scorer = FrameScorer(features_xy)
        initial = scorer.score(self.template, self.template_groups, (prior["x"], prior["y"], prior["yaw"]))
        fast = initial
        if prior_range <= self.maximum_dock_range_m:
            fast = refine_icp_with_prior(scorer, self.template, self.template_groups, prior, deadline=deadline)
            fast = refine_hypothesis_with_coarse_prior(
                scorer,
                self.template,
                self.template_groups,
                fast,
                prior,
                schedule=((0.5, 2.0), (0.1, 0.5), (0.025, 0.1)),
                max_iterations=2,
                deadline=deadline,
            )
        fast_geometry = float(fast.get("geometry_score", fast["score"]))
        fast_valid = (
            fast_geometry <= self.geometry_score_threshold
            and float(fast["support_normalized"]) >= self.minimum_geometry_support
            and int(fast["active_groups"]) >= self.minimum_active_groups
        )
        hypothesis_status = "UNIQUE"
        top = fast
        if not fast_valid and prior_range <= self.maximum_dock_range_m and time.perf_counter() < deadline:
            hypotheses, hypothesis_status = register_with_coarse_prior(
                features_xy, self.template, self.template_groups, prior, beam_width=3, deadline=deadline
            )
            if hypotheses:
                top = hypotheses[0]

        geometry_score = float(top.get("geometry_score", top["score"]))
        support = float(top["support_normalized"])
        valid = bool(
            prior_range <= self.maximum_dock_range_m
            and hypothesis_status == "UNIQUE"
            and geometry_score <= self.geometry_score_threshold
            and support >= self.minimum_geometry_support
            and int(top["active_groups"]) >= self.minimum_active_groups
        )
        confidence = pose_confidence(geometry_score, support, self.geometry_score_threshold) if valid else 0.0
        covariance = covariance_xyyaw(float(top["residual"]), confidence)
        refined_base = (float(top["x"]), float(top["y"]), float(top["yaw"]))
        refined_world = base_pose_to_world(*refined_base, vehicle)
        # 无效几何无法把回波分配到具体泊位，直接保持未知，避免浪费整个处理预算。
        occupancies = infer_bay_occupancy(
            points_level,
            refined_base,
            minimum_cells=self.occupancy_minimum_cells,
            ray_origins_base=ray_origins,
            ownship_mask=ownship_mask,
        ) if valid else [{"status": "UNKNOWN", "confidence": None, "cells": 0} for _ in range(3)]
        measurement_age = (self.get_clock().now().nanoseconds - int(snapshot["anchor"])) / 1e9
        if time.monotonic() - started > self.maximum_processing_s or not -.05 <= measurement_age <= self.output_stale_s:
            valid, confidence = False, 0.
        reason = "POINTCLOUD_GEOMETRY_VALIDATED" if valid else (
            "PROCESSING_OR_MEASUREMENT_DEADLINE" if measurement_age > self.output_stale_s or time.monotonic() - started > self.maximum_processing_s else
            "DOCK_OUT_OF_SENSOR_RANGE" if prior_range > self.maximum_dock_range_m else
            "DOCK_GEOMETRY_NOT_CONFIRMED"
        )
        local_candidates = make_berth_candidates(
            refined_base,
            occupancies,
            valid=valid,
            confidence=confidence,
            covariance=covariance,
            reason=reason,
            entrance_offset_m=self.entrance_offset_m,
        )
        world_candidates = make_berth_candidates(
            refined_world,
            occupancies,
            valid=valid,
            confidence=confidence,
            covariance=covariance,
            reason=reason,
            entrance_offset_m=self.entrance_offset_m,
        )
        common = {
            "stamp_ns": int(snapshot["anchor"]),
            "source_version": "phase17_berthing_perception_v1",
            "status": "READY" if valid else "DEGRADED",
            "global_id_resolved": True,
            "dock_geometry_score": geometry_score,
            "dock_geometry_support": support,
            "dock_active_groups": int(top["active_groups"]),
            "dock_prior_range_m": prior_range,
            "reason": reason,
            "input_mode": snapshot["input_mode"],
            "active_lidar_count": len(snapshot["clouds"]),
        }
        local_payload = {
            "schema": "team/berth_candidates_v2",
            "frame_id": self.base_frame,
            **common,
            "dock_pose_xyyaw_m_rad": [
                float(refined_base[0]), float(refined_base[1]), float(refined_base[2])
            ],
            "candidates": local_candidates,
        }
        world_payload = {
            "schema": "team/berth_candidates_world_v1",
            "frame_id": self.world_frame,
            "source_frame_id": self.base_frame,
            **common,
            "dock_pose_xyyaw_m_rad": [
                float(refined_world[0]), float(refined_world[1]), float(refined_world[2])
            ],
            "candidates": world_candidates,
        }
        self.local_publisher.publish(String(data=json.dumps(local_payload, ensure_ascii=False, separators=(",", ":"))))
        self.world_publisher.publish(String(data=json.dumps(world_payload, ensure_ascii=False, separators=(",", ":"))))

        pose_message = PoseWithCovarianceStamped()
        pose_message.header.stamp.sec = int(snapshot["anchor"] // 1_000_000_000)
        pose_message.header.stamp.nanosec = int(snapshot["anchor"] % 1_000_000_000)
        pose_message.header.frame_id = self.world_frame
        pose_message.pose.pose.position.x = refined_world[0]
        pose_message.pose.pose.position.y = refined_world[1]
        pose_message.pose.pose.position.z = float(dock.pose.position.z)
        qx, qy, qz, qw = yaw_to_quaternion(refined_world[2])
        pose_message.pose.pose.orientation.x = qx
        pose_message.pose.pose.orientation.y = qy
        pose_message.pose.pose.orientation.z = qz
        pose_message.pose.pose.orientation.w = qw
        pose_message.pose.covariance[0] = covariance[0]
        pose_message.pose.covariance[1] = covariance[1]
        pose_message.pose.covariance[5] = covariance[2]
        pose_message.pose.covariance[6] = covariance[3]
        pose_message.pose.covariance[7] = covariance[4]
        pose_message.pose.covariance[11] = covariance[5]
        pose_message.pose.covariance[30] = covariance[6]
        pose_message.pose.covariance[31] = covariance[7]
        pose_message.pose.covariance[35] = covariance[8]
        if valid:
            self.pose_publisher.publish(pose_message)

        elapsed = time.monotonic() - started
        with self.lock:
            self.last_output_stamp_ns = int(snapshot["anchor"])
            self.last_output_wall = time.monotonic()
            self.last_reason = reason
            self.last_input_issue = ""
            self.last_result = {
                "valid": valid,
                "geometry_score": geometry_score,
                "support": support,
                "active_groups": int(top["active_groups"]),
                "prior_range_m": prior_range,
                "refined_dock_base": list(refined_base),
                "refined_dock_world": list(refined_world),
                "occupancy": occupancies,
                "processing_s": elapsed,
                "raw_points": int(raw_points),
                "feature_cells": int(len(features_xy)),
                "sync_span_ms": float(snapshot["sync_span_ms"]),
                "input_mode": snapshot["input_mode"],
                "active_lidar_count": len(snapshot["clouds"]),
            }
            self.counts["OUTPUT_FRAMES"] += 1
            self.counts["VALID_FRAMES" if valid else "INVALID_FRAMES"] += 1

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                snapshot = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if snapshot is None:
                break
            try:
                self._process_snapshot(snapshot)
            except Exception as exc:
                with self.lock:
                    self.last_reason = f"PROCESSING_ERROR:{type(exc).__name__}"
                    self.counts[self.last_reason] += 1
                self.get_logger().error(f"靠泊帧处理失败: {exc}")

    def _publish_status(self) -> None:
        with self.lock:
            output_age = time.monotonic() - self.last_output_wall if self.last_output_wall else math.inf
            measurement_age = (self.get_clock().now().nanoseconds - self.last_output_stamp_ns) / 1e9 if self.last_output_stamp_ns else math.inf
            state = "READY" if (output_age <= 3. and -.05 <= measurement_age <= self.output_stale_s
                                and self.last_result and self.last_result["valid"]) else "DEGRADED"
            payload = {
                "schema": "team/berthing_status_v1",
                "status": state,
                "reason": self.last_reason,
                "last_input_issue": self.last_input_issue,
                "last_output_stamp_ns": self.last_output_stamp_ns,
                "output_age_wall_s": output_age if math.isfinite(output_age) else None,
                "measurement_age_sim_s": measurement_age if math.isfinite(measurement_age) else None,
                "inputs_seen": dict(self.counts),
                "last_result": self.last_result,
            }
        self.status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except Full:
            pass
        self.worker.join(timeout=10.0)
        if self.receipt_path:
            with self.lock:
                receipt = {
                    "schema": "phase17/berthing_perception_receipt_v1",
                    "started_wall_unix_s": self.started_wall,
                    "finished_wall_unix_s": time.time(),
                    "counts": dict(self.counts),
                    "last_reason": self.last_reason,
                    "last_input_issue": self.last_input_issue,
                    "last_result": self.last_result,
                    "production_truth_topics_subscribed": [],
                    "control_topics_published": [],
                }
            self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            self.receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    rclpy.init()
    node = BerthingPerceptionNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    interrupted = threading.Event()

    def stop_handler(*_) -> None:
        interrupted.set()
        executor.shutdown(timeout_sec=2.0)

    signal.signal(signal.SIGTERM, stop_handler)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        try:
            executor.remove_node(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
