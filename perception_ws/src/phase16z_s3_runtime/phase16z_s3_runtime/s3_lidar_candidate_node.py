"""五路 LaserScan 到 DynamicDetectionArray 的生产候选节点。

节点只读取五路 LaserScan、/clock 和固定 sensor->base TF，不订阅 GT、world
pose 或控制话题。每个 scan 保留自己的 header 时间戳，并且 TF 查询失败时
明确丢弃并记账；不使用 latest TF 伪造运动补偿。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import rclpy
from builtin_interfaces.msg import Time as RosTime
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros
from rclpy.time import Time

from phase16z_perception_msgs.msg import DynamicDetection, DynamicDetectionArray

from .s3_candidate_repair import (
    CandidateFunnel,
    CandidateObservation,
    cluster_base_scan,
    fuse_synchronized_candidates,
    load_footprint_yaml,
    scan_to_base_points,
)
from .s3_timestamped_frames import RigidTransform3D


SCAN_TOPICS = {
    "/wamv/sensors/lidars/back_lidar_sensor/scan": "back",
    "/wamv/sensors/lidars/forward_lidar_sensor/scan": "forward",
    "/wamv/sensors/lidars/front_lidar_sensor/scan": "front",
    "/wamv/sensors/lidars/left_lidar_sensor/scan": "left",
    "/wamv/sensors/lidars/right_lidar_sensor/scan": "right",
}
BASE_FRAME = "wamv/wamv/base_link"
DETECTIONS_TOPIC = "/team/perception/dynamic_detections_base"
STATUS_TOPIC = "/team/perception/status"
SYNC_TOLERANCE_NS = 100_000_000
MAX_QUEUE_ROWS = 64
MAX_CLUSTERS_PER_SCAN = 80


def stamp_ns(stamp: Any) -> int:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value <= 0:
        raise ValueError("ROS_HEADER_STAMP_INVALID")
    return value


def stamp_msg(value_ns: int) -> RosTime:
    result = RosTime()
    result.sec = int(value_ns // 1_000_000_000)
    result.nanosec = int(value_ns % 1_000_000_000)
    return result


def transform_yaw(transform: Any) -> float:
    rotation = transform.transform.rotation
    sin_yaw = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y)
    cos_yaw = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
    return math.atan2(sin_yaw, cos_yaw)


@dataclass
class ScanRow:
    row_id: int
    topic: str
    stamp_ns: int
    frame_id: str
    candidates: tuple[CandidateObservation, ...]
    tf_ok: bool
    error: str = ""


class S3LidarCandidateNode(Node):
    """读取五路真实 Scan，发布带逐观测时间的候选对象。"""

    def __init__(self) -> None:
        super().__init__("phase16z_s3_lidar_candidate_runtime")
        self.declare_parameter("receipt_path", "")
        self.declare_parameter("footprint_path", "")
        self.declare_parameter("tf_proof_path", "")
        self.receipt_path = Path(str(self.get_parameter("receipt_path").value).strip())
        footprint_path = str(self.get_parameter("footprint_path").value).strip()
        proof_path = str(self.get_parameter("tf_proof_path").value).strip()
        if not footprint_path:
            raise RuntimeError("SELF_FOOTPRINT_PATH_REQUIRED")
        if not proof_path:
            raise RuntimeError("FIXED_SENSOR_TF_PROOF_REQUIRED")
        self.footprint = load_footprint_yaml(footprint_path)
        self.tf_proof = self._load_tf_proof(proof_path)

        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.detections_publisher = self.create_publisher(DynamicDetectionArray, DETECTIONS_TOPIC, output_qos)
        self.status_publisher = self.create_publisher(String, STATUS_TOPIC, output_qos)
        self.input_subscriptions = [
            self.create_subscription(
                Clock,
                "/clock",
                self.clock_callback,
                QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT),
            )
        ]
        self.input_subscriptions.extend(
            self.create_subscription(
                LaserScan,
                topic,
                lambda message, topic_name=topic: self.scan_callback(message, topic_name),
                qos_profile_sensor_data,
            )
            for topic in SCAN_TOPICS
        )
        self.status_timer = self.create_timer(1.0, self.publish_status)

        self.started_monotonic = time.monotonic()
        self.scan_counts: dict[str, int] = defaultdict(int)
        self.scan_frames: dict[str, set[str]] = defaultdict(set)
        self.scan_drop_reasons: dict[str, int] = defaultdict(int)
        self.cluster_drop_reasons: dict[str, int] = defaultdict(int)
        self.sync_drop_reasons: dict[str, int] = defaultdict(int)
        self.base_drop_reasons: dict[str, int] = defaultdict(int)
        self.tf_attempts: dict[str, int] = defaultdict(int)
        self.tf_exact_successes: dict[str, int] = defaultdict(int)
        self.tf_exact_misses: dict[str, int] = defaultdict(int)
        self.tf_failures: dict[str, int] = defaultdict(int)
        self.tf_latest_rigid_fallbacks: dict[str, int] = defaultdict(int)
        self.scan_valid_points: dict[str, int] = defaultdict(int)
        self.scan_accepted_points: dict[str, int] = defaultdict(int)
        self.scan_range_max_m: dict[str, float] = {}
        self.funnel = CandidateFunnel()
        self.rows: dict[str, deque[ScanRow]] = {topic: deque() for topic in SCAN_TOPICS}
        self.next_row_id = 1
        self.used_rows: set[int] = set()
        self.groups_formed = 0
        self.groups_with_candidates = 0
        self.async_groups = 0
        self.group_audit: list[dict[str, Any]] = []
        self.reused_selected_rows = 0
        self.base_detection_messages = 0
        self.base_detection_objects = 0
        self.base_detection_nonempty_messages = 0
        self.base_detection_empty_messages = 0
        self.clock_count = 0
        self.clock_first_ns: int | None = None
        self.clock_last_ns: int | None = None
        self.clock_regressions = 0
        self.application_queue_max = 0

    @staticmethod
    def _load_tf_proof(path: str) -> dict[str, Any]:
        proof_path = Path(path)
        if not proof_path.exists():
            raise RuntimeError(f"FIXED_SENSOR_TF_PROOF_NOT_FOUND:{proof_path}")
        try:
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("FIXED_SENSOR_TF_PROOF_INVALID_JSON") from exc
        required = ("model_name", "model_source", "model_sha256", "tf_chain", "sdf_sha256")
        if not isinstance(proof, dict) or any(not proof.get(key) for key in required):
            raise RuntimeError("FIXED_SENSOR_TF_PROOF_FIELDS_MISSING")
        if not isinstance(proof["tf_chain"], dict) or any(
            not proof["tf_chain"].get(label) for label in SCAN_TOPICS.values()
        ):
            raise RuntimeError("FIXED_SENSOR_TF_PROOF_SCAN_CHAIN_MISSING")
        return proof

    def clock_callback(self, message: Clock) -> None:
        value = int(message.clock.sec) * 1_000_000_000 + int(message.clock.nanosec)
        self.clock_count += 1
        if self.clock_last_ns is not None and value < self.clock_last_ns:
            self.clock_regressions += 1
        self.clock_last_ns = value
        if self.clock_first_ns is None:
            self.clock_first_ns = value

    def _lookup_sensor_tf(self, source_frame: str, stamp: int, topic: str) -> Any:
        if not source_frame:
            raise ValueError("SENSOR_FRAME_MISSING")
        self.tf_attempts[topic] += 1
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                source_frame,
                Time(nanoseconds=int(stamp)),
            )
        except Exception as exc:
            self.tf_exact_misses[topic] += 1
            # 只对已经由同一 WSL 模型、SDF 哈希和 TF chain 证明为固定的
            # sensor->base 外参使用 latest 查询；运动中的 ENU->base 不经过这里。
            try:
                rigid_transform = self.tf_buffer.lookup_transform(
                    BASE_FRAME,
                    source_frame,
                    Time(),
                )
            except Exception as latest_exc:
                self.tf_failures[topic] += 1
                raise ValueError("SENSOR_TO_BASE_TF_AT_OBSERVATION_TIME_UNAVAILABLE") from latest_exc
            if (
                str(rigid_transform.header.frame_id).strip() != BASE_FRAME
                or str(rigid_transform.child_frame_id).strip() != source_frame
            ):
                self.tf_failures[topic] += 1
                raise ValueError("SENSOR_TO_BASE_TF_CHAIN_MISMATCH") from exc
            self.tf_latest_rigid_fallbacks[topic] += 1
            transform = rigid_transform
        if str(transform.header.frame_id).strip() != BASE_FRAME or str(transform.child_frame_id).strip() != source_frame:
            self.tf_failures[topic] += 1
            raise ValueError("SENSOR_TO_BASE_TF_CHAIN_MISMATCH")
        self.tf_exact_successes[topic] += 1
        return transform

    def _make_candidate_row(self, message: LaserScan, topic: str) -> ScanRow:
        stamp = stamp_ns(message.header.stamp)
        frame_id = str(message.header.frame_id).strip()
        transform = self._lookup_sensor_tf(frame_id, stamp, topic)
        rigid = RigidTransform3D(
            translation_x_m=float(transform.transform.translation.x),
            translation_y_m=float(transform.transform.translation.y),
            translation_z_m=float(transform.transform.translation.z),
            quaternion_x=float(transform.transform.rotation.x),
            quaternion_y=float(transform.transform.rotation.y),
            quaternion_z=float(transform.transform.rotation.z),
            quaternion_w=float(transform.transform.rotation.w),
            source_frame=frame_id,
            target_frame=BASE_FRAME,
            is_model_proven_fixed=True,
        )
        local_funnel = CandidateFunnel()
        points = scan_to_base_points(
            stamp_ns=stamp,
            source_sensor=SCAN_TOPICS[topic],
            ranges_m=message.ranges,
            angle_min_rad=float(message.angle_min),
            angle_increment_rad=float(message.angle_increment),
            range_min_m=float(message.range_min),
            range_max_m=float(message.range_max),
            sensor_to_base=rigid,
            self_footprint=self.footprint,
            funnel=local_funnel,
        )
        self.funnel.input_points += local_funnel.input_points
        self.funnel.accepted_points += local_funnel.accepted_points
        self.funnel.transformed_points += local_funnel.transformed_points
        self.funnel.dropped_nonfinite += local_funnel.dropped_nonfinite
        self.funnel.dropped_sensor_range += local_funnel.dropped_sensor_range
        self.funnel.dropped_self_body += local_funnel.dropped_self_body
        # 安全地合并drop_reasons
        for key, value in local_funnel.drop_reasons.items():
            self.funnel.drop_reasons[key] += value
        self.scan_valid_points[topic] += local_funnel.input_points - local_funnel.dropped_nonfinite - local_funnel.dropped_sensor_range
        self.scan_accepted_points[topic] += len(points)
        candidates = cluster_base_scan(
            points,
            minimum_support=2,
            range_jump_m=0.75,  # 实验D：恢复baseline，只测试Self-filter
            point_gap_m=1.25,   # 实验D：恢复baseline
            funnel=local_funnel,
        )
        # 安全地合并聚类drop_reasons
        for key, value in local_funnel.drop_reasons.items():
            self.cluster_drop_reasons[key] += value
        if len(candidates) > MAX_CLUSTERS_PER_SCAN:
            candidates = candidates[:MAX_CLUSTERS_PER_SCAN]
            self.cluster_drop_reasons["cluster_limit_reached"] += 1
        return ScanRow(self.next_row_id, topic, stamp, frame_id, candidates, True)

    def scan_callback(self, message: LaserScan, topic: str) -> None:
        self.scan_counts[topic] += 1
        frame_id = str(message.header.frame_id).strip()
        if frame_id:
            self.scan_frames[topic].add(frame_id)
        self.scan_range_max_m[topic] = float(message.range_max)
        try:
            row = self._make_candidate_row(message, topic)
        except (TypeError, ValueError) as exc:
            stamp = 0
            try:
                stamp = stamp_ns(message.header.stamp)
            except (TypeError, ValueError):
                self.scan_drop_reasons["ROS_HEADER_STAMP_INVALID"] += 1
            row = ScanRow(self.next_row_id, topic, stamp, frame_id, (), False, str(exc))
            self.scan_drop_reasons[str(exc)] += 1
        self.next_row_id += 1
        if row.stamp_ns > 0:
            self.rows[topic].append(row)
        self.application_queue_max = max(self.application_queue_max, sum(len(queue) for queue in self.rows.values()))
        while sum(len(queue) for queue in self.rows.values()) > MAX_QUEUE_ROWS * len(SCAN_TOPICS):
            oldest_topic = min(SCAN_TOPICS, key=lambda item: self.rows[item][0].stamp_ns if self.rows[item] else 2**63)
            if self.rows[oldest_topic]:
                self.rows[oldest_topic].popleft()
                self.sync_drop_reasons["bounded_queue_overflow"] += 1
        self.try_form_groups()

    def _find_match(self, topic: str, stamp: int) -> ScanRow | None:
        choices = [row for row in self.rows[topic] if abs(int(row.stamp_ns) - stamp) <= SYNC_TOLERANCE_NS]
        return min(choices, key=lambda row: abs(int(row.stamp_ns) - stamp), default=None)

    def try_form_groups(self) -> None:
        primary_topic = next(iter(SCAN_TOPICS))
        primary_queue = self.rows[primary_topic]
        while primary_queue:
            primary = primary_queue[0]
            matches: dict[str, ScanRow] = {primary_topic: primary}
            missing: list[str] = []
            for topic in SCAN_TOPICS:
                if topic == primary_topic:
                    continue
                match = self._find_match(topic, primary.stamp_ns)
                if match is None:
                    missing.append(topic)
                else:
                    matches[topic] = match
            if missing:
                latest_seen = max((queue[-1].stamp_ns for queue in self.rows.values() if queue), default=primary.stamp_ns)
                if latest_seen - primary.stamp_ns <= SYNC_TOLERANCE_NS:
                    break
                primary_queue.popleft()
                self.sync_drop_reasons["missing_after_bounded_wait:" + ",".join(SCAN_TOPICS[item] for item in missing)] += 1
                continue
            # The entire five-stream group, not just each primary offset,
            # must fit the same span budget. Drop one incompatible selected row
            # to make progress; never publish a group beyond the budget.
            selected_stamps = [int(row.stamp_ns) for row in matches.values()]
            if max(selected_stamps) - min(selected_stamps) > SYNC_TOLERANCE_NS:
                oldest_topic = min(matches, key=lambda topic: (matches[topic].stamp_ns, topic))
                self.rows[oldest_topic].remove(matches[oldest_topic])
                self.sync_drop_reasons["group_span_exceeded"] += 1
                continue
            # Remove the selected object from its own queue even if it is not
            # the head. Retaining non-head matches reuses sensor observations.
            for topic, row in matches.items():
                self.rows[topic].remove(row)
            for queue in self.rows.values():
                while queue and queue[0].stamp_ns < primary.stamp_ns - SYNC_TOLERANCE_NS:
                    queue.popleft()
                    self.sync_drop_reasons["stale_unmatched_row_evicted"] += 1
            self._emit_group(tuple(matches.values()), primary.stamp_ns)

    def _emit_group(self, rows: tuple[ScanRow, ...], primary_stamp: int) -> None:
        self.groups_formed += 1
        row_ids = {SCAN_TOPICS[topic]: int(row.row_id) for topic, row in ((item.topic, item) for item in rows)}
        row_stamps = {SCAN_TOPICS[topic]: int(row.stamp_ns) for topic, row in ((item.topic, item) for item in rows)}
        group_span_ns = max(row_stamps.values()) - min(row_stamps.values())
        reused = any(row.row_id in self.used_rows for row in rows)
        if reused:
            self.reused_selected_rows += 1
        self.used_rows.update(row.row_id for row in rows)
        self.group_audit.append(
            {
                "group_index": int(self.groups_formed),
                "primary_stamp_ns": int(primary_stamp),
                "row_ids": row_ids,
                "row_stamps_ns": row_stamps,
                "span_ns": int(group_span_ns),
                "selected_rows_reused": bool(reused),
                "queue_depth_after_selection": int(sum(len(queue) for queue in self.rows.values())),
                "consumed": True,
            }
        )
        candidates = tuple(item for row in rows for item in row.candidates)
        self.funnel.clusters_seen += 0
        self.funnel.candidates_emitted += len(candidates)
        if not candidates:
            self.base_drop_reasons["NO_CLUSTER_CANDIDATE_AFTER_SYNC"] += 1
            output_candidates: tuple[CandidateObservation, ...] = ()
        elif len({int(row.stamp_ns) for row in rows}) == 1:
            output_candidates = fuse_synchronized_candidates(candidates, funnel=self.funnel)
        else:
            self.async_groups += 1
            self.sync_drop_reasons["asynchronous_group_deferred_to_timestamped_enu_fusion"] += 1
            output_candidates = candidates
        if output_candidates:
            self.groups_with_candidates += 1
        output = DynamicDetectionArray()
        output.header.stamp = stamp_msg(int(primary_stamp))
        output.header.frame_id = BASE_FRAME
        for candidate in output_candidates:
            detection = DynamicDetection()
            detection.header.stamp = stamp_msg(int(candidate.stamp_ns))
            detection.header.frame_id = BASE_FRAME
            detection.observation_id = candidate.observation_id
            detection.center.x = float(candidate.x_m)
            detection.center.y = float(candidate.y_m)
            detection.center.z = 0.0
            detection.radius_m = float(candidate.radius_m)
            detection.support = int(candidate.support)
            detection.position_std_m = float(candidate.position_std_m)
            detection.range_m = float(candidate.range_m)
            detection.confidence = float(candidate.confidence)
            detection.source_sensors = list(candidate.source_sensors)
            output.detections.append(detection)
        self.detections_publisher.publish(output)
        self.base_detection_messages += 1
        self.base_detection_objects += len(output_candidates)
        if output_candidates:
            self.base_detection_nonempty_messages += 1
        else:
            self.base_detection_empty_messages += 1

    def build_receipt(self, status: str) -> dict[str, Any]:
        receipt = {
            "schema": "phase16z_s3_lidar_candidate_runtime_v2",
            "status": status,
            "base_frame": BASE_FRAME,
            "detections_topic": DETECTIONS_TOPIC,
            "scan_topics": list(SCAN_TOPICS),
            "input_funnel": {
                "five_scan_messages": {topic: int(self.scan_counts[topic]) for topic in SCAN_TOPICS},
                "five_scan_messages_total": int(sum(self.scan_counts.values())),
                "sync_groups": int(self.groups_formed),
                "sync_group_width": len(SCAN_TOPICS),
                "cluster_candidates": int(self.funnel.candidates_emitted),
                "base_detection_objects": int(self.base_detection_objects),
                "base_detection_messages": int(self.base_detection_messages),
                "base_detection_nonempty_messages": int(self.base_detection_nonempty_messages),
                "base_detection_empty_messages": int(self.base_detection_empty_messages),
                "drop_reasons": {
                    "scan": dict(sorted(self.scan_drop_reasons.items())),
                    "sync": dict(sorted(self.sync_drop_reasons.items())),
                    "cluster": dict(sorted(self.cluster_drop_reasons.items())),
                    "base_detection": dict(sorted(self.base_drop_reasons.items())),
                },
            },
            "candidate_core_funnel": self.funnel.as_dict(),
            "scan_frames": {topic: sorted(self.scan_frames[topic]) for topic in SCAN_TOPICS},
            "scan_valid_points": {topic: int(self.scan_valid_points[topic]) for topic in SCAN_TOPICS},
            "scan_accepted_points": {topic: int(self.scan_accepted_points[topic]) for topic in SCAN_TOPICS},
            "scan_range_max_m": {
                topic: float(self.scan_range_max_m.get(topic, 0.0)) for topic in SCAN_TOPICS
            },
            "sensor_tf": {
                "attempts": {topic: int(self.tf_attempts[topic]) for topic in SCAN_TOPICS},
                "exact_successes": {topic: int(self.tf_exact_successes[topic]) for topic in SCAN_TOPICS},
                "exact_lookup_misses_recovered": {topic: int(self.tf_exact_misses[topic]) for topic in SCAN_TOPICS},
                "failures": {topic: int(self.tf_failures[topic]) for topic in SCAN_TOPICS},
                "latest_rigid_fallbacks": {topic: int(self.tf_latest_rigid_fallbacks[topic]) for topic in SCAN_TOPICS},
            },
            "clock": {
                "message_count": int(self.clock_count),
                "first_stamp_ns": self.clock_first_ns,
                "last_stamp_ns": self.clock_last_ns,
                "regressions": int(self.clock_regressions),
            },
            "async_groups_deferred_to_enu": int(self.async_groups),
            "application_queue_max": int(self.application_queue_max),
            "sync_summary": {
                "groups_formed": int(self.groups_formed),
                "selected_rows": int(len(self.used_rows)),
                "selected_row_reuse_groups": int(self.reused_selected_rows),
                "max_group_span_ns": max((int(item["span_ns"]) for item in self.group_audit), default=0),
                "audit_complete": status == "MEASURED",
            },
            "footprint": {"source": self.footprint.source, "component_count": self.footprint.component_count},
            "fixed_sensor_tf_proof": self.tf_proof,
            "forbidden_topics_subscribed": [],
            "gt_input_subscription_count": 0,
            "world_input_subscription_count": 0,
            "actuator_publishers": 0,
            "elapsed_wall_s": time.monotonic() - self.started_monotonic,
        }

        if status == "MEASURED":
            receipt["sync_group_audit"] = list(self.group_audit)
        return receipt

    def publish_status(self) -> None:
        message = String()
        message.data = json.dumps(self.build_receipt("RUNNING"), ensure_ascii=False, sort_keys=True)
        self.status_publisher.publish(message)

    def write_receipt(self) -> None:
        if not self.receipt_path:
            return
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_path.write_text(
            json.dumps(self.build_receipt("MEASURED"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: S3LidarCandidateNode | None = None
    try:
        node = S3LidarCandidateNode()
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        if node is not None:
            node.write_receipt()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
