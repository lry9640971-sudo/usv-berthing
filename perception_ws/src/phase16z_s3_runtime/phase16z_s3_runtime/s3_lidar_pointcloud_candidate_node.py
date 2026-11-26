"""五路 PointCloud2 到 DynamicDetectionArray 的三维生产候选节点。"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from queue import Full, Queue
import signal
import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
import rclpy
import tf2_ros
from rclpy.time import Time

from phase16z_perception_msgs.msg import DynamicDetection, DynamicDetectionArray

from .s3_candidate_repair import (
    CandidateObservation,
    FootprintUnion,
    fuse_synchronized_candidates,
    load_footprint_yaml,
)
from .s3_lidar_candidate_node import BASE_FRAME, DETECTIONS_TOPIC, STATUS_TOPIC, stamp_msg, stamp_ns
from .s3_pointcloud_repair import (
    PointCloudFunnel,
    cluster_base_pointcloud,
    organized_point_features,
    organized_vertical_neighbor_counts,
    pointcloud_to_base_points,
)
from .s3_partial_sync import pop_ready_group
from .s3_fair_cloud_queue import FairLatestCloudMailbox
from .s3_ownship_geometry import load_ownship_geometry
from .s3_timestamped_frames import RigidTransform3D, PlanarPose, TimestampedPoseBuffer


POINTCLOUD_TOPICS = {
    "/wamv/sensors/lidars/back_lidar_sensor/points": "back",
    "/wamv/sensors/lidars/forward_lidar_sensor/points": "forward",
    "/wamv/sensors/lidars/front_lidar_sensor/points": "front",
    "/wamv/sensors/lidars/left_lidar_sensor/points": "left",
    "/wamv/sensors/lidars/right_lidar_sensor/points": "right",
}
SYNC_TOLERANCE_NS = 100_000_000
MAX_QUEUE_ROWS = 16


@dataclass
class CloudRow:
    row_id: int
    topic: str
    stamp_ns: int
    frame_id: str
    candidates: tuple[CandidateObservation, ...]
    point_count: int
    finite_point_count: int
    error: str = ""
    points: tuple = ()


class S3LidarPointCloudCandidateNode(Node):
    """消费完整 XYZ 点云，生产带三维审计信息的候选对象。"""

    def __init__(self) -> None:
        super().__init__("phase16z_s3_lidar_pointcloud_candidate_runtime")
        self.declare_parameter("receipt_path", "")
        self.declare_parameter("footprint_path", "")
        self.declare_parameter("tf_proof_path", "")
        self.declare_parameter("voxel_size_m", 0.50)
        self.declare_parameter("cluster_radius_m", 0.90)
        # 双通道中完整候选需要较小半径分开相邻船，结构证据需要较大半径
        # 汇聚同一船体的多层点。分别配置，避免一组半径同时损害召回和纯度。
        self.declare_parameter("structural_voxel_size_m", 0.50)
        self.declare_parameter("structural_cluster_radius_m", 0.90)
        # 五路点云合并后，后侧远距离船常只有单个有效体素；轨迹成熟度和
        # 水面净高在下游承担稳健过滤，不能在候选入口把这些船永久删除。
        self.declare_parameter("minimum_support", 1)
        self.declare_parameter("maximum_clusters", 400)
        self.declare_parameter("ownship_z_min_m", -1.0)
        self.declare_parameter("ownship_z_max_m", 3.0)
        self.declare_parameter("detection_z_min_m", -0.5)
        self.declare_parameter("detection_z_max_m", 1.5)
        self.declare_parameter("maximum_detection_range_m", 80.0)
        self.declare_parameter("sync_tolerance_ms", 100.0)
        self.declare_parameter("defer_sync", False)
        self.declare_parameter("level_height_filter", False)
        self.declare_parameter("cloud_water_plane_level_filter", False)
        self.declare_parameter("water_surface_z_m", 0.05)
        self.declare_parameter("water_clearance_m", 0.07)
        self.declare_parameter("organized_structure_only", False)
        self.declare_parameter("organized_dual_path", False)
        self.declare_parameter("dual_path_evidence_merge_distance_m", 1.5)
        self.declare_parameter("maximum_vertical_range_delta_m", 0.30)
        self.declare_parameter("maximum_vertical_surface_normal_up_abs", 0.70)
        self.declare_parameter("minimum_vertical_consistent_neighbors", 1)
        self.declare_parameter("cluster_center_mode", "mean")
        self.declare_parameter("cluster_center_radial_offset_m", 0.0)
        self.level_height_filter = bool(self.get_parameter("level_height_filter").value)
        self.cloud_water_plane_level_filter = bool(
            self.get_parameter("cloud_water_plane_level_filter").value
        )
        if self.cloud_water_plane_level_filter and not self.level_height_filter:
            raise RuntimeError("CLOUD_WATER_PLANE_FILTER_REQUIRES_LEVEL_HEIGHT_FILTER")
        self.water_surface_z_m = float(self.get_parameter("water_surface_z_m").value)
        self.water_clearance_m = float(self.get_parameter("water_clearance_m").value)
        self.organized_structure_only = bool(self.get_parameter("organized_structure_only").value)
        self.organized_dual_path = bool(self.get_parameter("organized_dual_path").value)
        self.dual_path_evidence_merge_distance_m = float(
            self.get_parameter("dual_path_evidence_merge_distance_m").value
        )
        self.maximum_vertical_range_delta_m = float(self.get_parameter("maximum_vertical_range_delta_m").value)
        self.maximum_vertical_surface_normal_up_abs = float(
            self.get_parameter("maximum_vertical_surface_normal_up_abs").value
        )
        self.minimum_vertical_consistent_neighbors = int(
            self.get_parameter("minimum_vertical_consistent_neighbors").value
        )
        self.cluster_center_mode = str(self.get_parameter("cluster_center_mode").value).strip()
        self.cluster_center_radial_offset_m = float(
            self.get_parameter("cluster_center_radial_offset_m").value
        )
        receipt_text = str(self.get_parameter("receipt_path").value).strip()
        self.receipt_path = Path(receipt_text) if receipt_text else None
        footprint_path = str(self.get_parameter("footprint_path").value).strip()
        proof_path = str(self.get_parameter("tf_proof_path").value).strip()
        if not footprint_path or not proof_path:
            raise RuntimeError("POINTCLOUD_FOOTPRINT_AND_TF_PROOF_REQUIRED")
        self.footprint: FootprintUnion = load_footprint_yaml(footprint_path)
        self.self_geometry = load_ownship_geometry(Path(footprint_path).with_name("visual_self_geometry.yaml"))
        self.tf_proof = self._load_tf_proof(proof_path)
        self.voxel_size_m = float(self.get_parameter("voxel_size_m").value)
        self.cluster_radius_m = float(self.get_parameter("cluster_radius_m").value)
        self.structural_voxel_size_m = float(
            self.get_parameter("structural_voxel_size_m").value
        )
        self.structural_cluster_radius_m = float(
            self.get_parameter("structural_cluster_radius_m").value
        )
        self.minimum_support = int(self.get_parameter("minimum_support").value)
        self.maximum_clusters = int(self.get_parameter("maximum_clusters").value)
        self.ownship_z_min_m = float(self.get_parameter("ownship_z_min_m").value)
        self.ownship_z_max_m = float(self.get_parameter("ownship_z_max_m").value)
        self.detection_z_min_m = float(self.get_parameter("detection_z_min_m").value)
        self.detection_z_max_m = float(self.get_parameter("detection_z_max_m").value)
        self.maximum_detection_range_m = float(self.get_parameter("maximum_detection_range_m").value)
        self.sync_tolerance_ns = int(float(self.get_parameter("sync_tolerance_ms").value) * 1e6)
        self.defer_sync = bool(self.get_parameter("defer_sync").value)
        if not all(math.isfinite(value) for value in (
            self.voxel_size_m, self.cluster_radius_m, self.ownship_z_min_m,
            self.structural_voxel_size_m, self.structural_cluster_radius_m,
            self.ownship_z_max_m, self.detection_z_min_m, self.detection_z_max_m,
            self.maximum_detection_range_m,
            self.dual_path_evidence_merge_distance_m,
            self.cluster_center_radial_offset_m,
        )):
            raise RuntimeError("POINTCLOUD_PARAMETER_NONFINITE")
        if (
            self.voxel_size_m <= 0.0
            or self.cluster_radius_m <= 0.0
            or self.structural_voxel_size_m <= 0.0
            or self.structural_cluster_radius_m <= 0.0
            or self.minimum_support <= 0
            or self.maximum_detection_range_m <= 0.0
            or self.sync_tolerance_ns <= 0
            or self.dual_path_evidence_merge_distance_m <= 0.0
            or self.cluster_center_mode not in {"mean", "pca_bbox"}
        ):
            raise RuntimeError("POINTCLOUD_PARAMETER_INVALID")
        if self.detection_z_min_m >= self.detection_z_max_m:
            raise RuntimeError("POINTCLOUD_DETECTION_Z_RANGE_INVALID")

        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self.callback_group = ReentrantCallbackGroup()
        self.rows_lock = threading.RLock()
        self.height_lock = threading.RLock()
        self.height_pose_buffer = TimestampedPoseBuffer()
        self.pending_height_clouds = deque()
        self.height_latest_stamp = None
        output_qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        # 离线排空保留深队列；在线只保留最新测量，DDS 不能先积压 128 帧再进入邮箱。
        cloud_qos = QoSProfile(depth=128 if self.defer_sync else 2,
            reliability=ReliabilityPolicy.RELIABLE if self.defer_sync else ReliabilityPolicy.BEST_EFFORT)
        self.detections_publisher = self.create_publisher(DynamicDetectionArray, DETECTIONS_TOPIC, output_qos)
        self.status_publisher = self.create_publisher(String, STATUS_TOPIC, output_qos)
        self.input_subscriptions = [
            self.create_subscription(Clock, "/clock", self.clock_callback, qos_profile_sensor_data, callback_group=self.callback_group),
        ]
        if self.level_height_filter:
            self.input_subscriptions.append(self.create_subscription(
                Odometry, "/team/localization/odometry", self.odometry_callback, 100,
                callback_group=self.callback_group))
        self.input_subscriptions.extend(
            self.create_subscription(
                PointCloud2,
                topic,
                lambda message, topic_name=topic: self.cloud_callback(message, topic_name),
                cloud_qos,
                callback_group=self.callback_group,
            )
            for topic in POINTCLOUD_TOPICS
        )
        self.status_timer = self.create_timer(1.0, self.publish_status, callback_group=self.callback_group)

        self.started_monotonic = time.monotonic()
        self.cloud_counts: dict[str, int] = defaultdict(int)
        self.cloud_frames: dict[str, set[str]] = defaultdict(set)
        self.cloud_shape: dict[str, dict[str, Any]] = {}
        self.cloud_point_counts: dict[str, int] = defaultdict(int)
        self.cloud_finite_counts: dict[str, int] = defaultdict(int)
        self.cloud_funnels: dict[str, PointCloudFunnel] = {
            topic: PointCloudFunnel() for topic in POINTCLOUD_TOPICS
        }
        self.cloud_drop_reasons: dict[str, int] = defaultdict(int)
        self.tf_attempts: dict[str, int] = defaultdict(int)
        self.tf_exact_misses: dict[str, int] = defaultdict(int)
        self.tf_exact_successes: dict[str, int] = defaultdict(int)
        self.tf_latest_fallbacks: dict[str, int] = defaultdict(int)
        self.tf_failures: dict[str, int] = defaultdict(int)
        self.sync_drop_reasons: dict[str, int] = defaultdict(int)
        self.base_drop_reasons: dict[str, int] = defaultdict(int)
        self.funnel = PointCloudFunnel()
        self.rows: dict[str, deque[CloudRow]] = {topic: deque() for topic in POINTCLOUD_TOPICS}
        # 点云解码主要受 Python/GIL 限制。五个 worker 同时解码会互相争用并让
        # 不同雷达的处理时间轴分叉；单个有界队列按到达顺序处理，在实测中既
        # 更快，也让同步器看到完整的五路相邻时间戳。
        # 实时输出必须优先于离线逐帧完整性。大点云处理来不及时只保留最近
        # 两个五路周期，避免规划收到数秒前的目标位置。
        self.pending_clouds = FairLatestCloudMailbox(POINTCLOUD_TOPICS)
        self.pending_clouds_lock = threading.Lock()
        self.worker_queue_peak = 0
        self.worker_stop = threading.Event()
        self.worker_thread = threading.Thread(
            target=self._cloud_worker, name="pointcloud-ordered", daemon=True)
        self.worker_thread.start()
        self.workers_finished = False
        self.next_row_id = 1
        self.groups_formed = 0
        self.group_sensor_count_histogram: dict[int, int] = defaultdict(int)
        self.last_emitted_group_stamp_ns = 0
        self.groups_with_candidates = 0
        self.base_detection_messages = 0
        self.base_detection_objects = 0
        self.base_detection_nonempty_messages = 0
        self.base_detection_empty_messages = 0
        self.clock_count = 0
        self.clock_first_ns: int | None = None
        self.clock_last_ns: int | None = None
        self.clock_regressions = 0

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
        if not isinstance(proof["tf_chain"], dict) or any(not proof["tf_chain"].get(label) for label in POINTCLOUD_TOPICS.values()):
            raise RuntimeError("FIXED_SENSOR_TF_PROOF_POINTCLOUD_CHAIN_MISSING")
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
            raise ValueError("POINTCLOUD_FRAME_MISSING")
        self.tf_attempts[topic] += 1
        try:
            transform = self.tf_buffer.lookup_transform(BASE_FRAME, source_frame, Time(nanoseconds=int(stamp)))
        except Exception as exact_exc:
            try:
                transform = self.tf_buffer.lookup_transform(BASE_FRAME, source_frame, Time())
            except Exception as latest_exc:
                self.tf_failures[topic] += 1
                raise ValueError("SENSOR_TO_BASE_TF_UNAVAILABLE") from latest_exc
            self.tf_exact_misses[topic] += 1
            self.tf_latest_fallbacks[topic] += 1
            if str(transform.header.frame_id).strip() != BASE_FRAME or str(transform.child_frame_id).strip() != source_frame:
                self.tf_failures[topic] += 1
                raise ValueError("SENSOR_TO_BASE_TF_CHAIN_MISMATCH") from exact_exc
        if str(transform.header.frame_id).strip() != BASE_FRAME or str(transform.child_frame_id).strip() != source_frame:
            self.tf_failures[topic] += 1
            raise ValueError("SENSOR_TO_BASE_TF_CHAIN_MISMATCH")
        self.tf_exact_successes[topic] += 1
        return transform

    def odometry_callback(self, message: Odometry) -> None:
        if message.header.frame_id != "team_enu" or message.child_frame_id != BASE_FRAME:
            self.cloud_drop_reasons["HEIGHT_POSE_FRAME_INVALID"] += 1
            return
        p, q = message.pose.pose.position, message.pose.pose.orientation
        stamp = stamp_ns(message.header.stamp)
        quaternion = (q.x, q.y, q.z, q.w)
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        with self.height_lock:
            self.height_pose_buffer.add(PlanarPose(stamp,p.x,p.y,yaw,z_m=p.z,quaternion_xyzw=quaternion))
            self.height_latest_stamp = stamp
            ready = []
            pending = deque()
            for cloud, topic in self.pending_height_clouds:
                (ready if stamp_ns(cloud.header.stamp) <= stamp else pending).append((cloud,topic))
            self.pending_height_clouds = pending
        for cloud, topic in ready:
            self.pending_clouds.put(topic, cloud)

    def _make_cloud_row(self, message: PointCloud2, topic: str) -> CloudRow:
        phase_started = time.perf_counter()
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
        tf_finished = time.perf_counter()
        # 使用 ROS2 的 NumPy 解码路径。逐点 structured-array 访问在 WSL
        # 的五路 360x48 点云上会把 worker 拖到数秒/帧，导致同步器只能
        # 发出单雷达部分组，动态船在下一帧到来前已经失去关联。
        # 保留 NaN/Inf，由统一 funnel 负责统计和过滤。
        raw = point_cloud2.read_points(message, field_names=["x", "y", "z"], skip_nans=False)
        xyz = np.column_stack((raw["x"], raw["y"], raw["z"]))
        organized_normal_up, vertical_range_delta = organized_point_features(
            xyz, width=int(message.width), height=int(message.height), sensor_to_base=rigid)
        vertical_neighbor_count = organized_vertical_neighbor_counts(
            xyz, width=int(message.width), height=int(message.height))
        features_finished = time.perf_counter()
        local_funnel = PointCloudFunnel()
        up_base = None
        height_origin = 0.0
        height_min = self.detection_z_min_m
        if self.level_height_filter:
            with self.height_lock:
                pose = self.height_pose_buffer.interpolate(stamp)
            up_base = tuple(Rotation.from_quat(pose.quaternion_xyzw).as_matrix()[2])
            height_origin = pose.z_m - self.water_surface_z_m
            height_min = self.water_clearance_m
        points = pointcloud_to_base_points(
            stamp_ns=stamp,
            source_sensor=POINTCLOUD_TOPICS[topic],
            # 把完整数组交给 funnel，由统一入口统计非有限点，避免 receipt
            # 把“节点预过滤后的数量”误报成原始点数。
            points_xyz_m=xyz,
            sensor_to_base=rigid,
            self_footprint=self.footprint,
            self_geometry=self.self_geometry,
            ownship_z_min_m=self.ownship_z_min_m,
            ownship_z_max_m=self.ownship_z_max_m,
            detection_z_min_m=height_min,
            detection_z_max_m=self.detection_z_max_m,
            maximum_detection_range_m=self.maximum_detection_range_m,
            organized_normal_up_abs=organized_normal_up,
            min_vertical_range_delta_m=vertical_range_delta,
            vertical_consistent_neighbor_count=vertical_neighbor_count,
            gravity_up_base=up_base,
            height_origin_m=height_origin,
            estimate_height_origin_from_cloud=self.cloud_water_plane_level_filter,
            funnel=local_funnel,
        )
        self.last_cloud_phases = {"sensor": POINTCLOUD_TOPICS[topic], "tf_s": tf_finished-phase_started,
            "decode_features_s": features_finished-tf_finished,
            "point_filter_s": time.perf_counter()-features_finished,
            "input_age_sim_s": (self.get_clock().now().nanoseconds-stamp)/1e9}
        # 先保留各雷达稀疏点，同步后再聚类；避免每路单独做点数门丢失目标。
        candidates = ()
        self.funnel.merge(local_funnel)
        self.cloud_funnels[topic].merge(local_funnel)
        self.cloud_point_counts[topic] += len(raw)
        self.cloud_finite_counts[topic] += local_funnel.finite_points
        self.cloud_frames[topic].add(frame_id)
        self.cloud_shape[topic] = {
            "width": int(message.width),
            "height": int(message.height),
            "point_step": int(message.point_step),
            "row_step": int(message.row_step),
            "is_bigendian": bool(message.is_bigendian),
            "is_dense": bool(message.is_dense),
            "fields": [
                {"name": str(field.name), "offset": int(field.offset), "datatype": int(field.datatype), "count": int(field.count)}
                for field in message.fields
            ],
        }
        return CloudRow(self.next_row_id, topic, stamp, frame_id, candidates, len(raw), local_funnel.finite_points,
                        points=tuple(points))

    def cloud_callback(self, message: PointCloud2, topic: str) -> None:
        self.cloud_counts[topic] += 1
        with self.pending_clouds_lock:
            if self.pending_clouds.put(topic, message):
                self.cloud_drop_reasons["worker_same_sensor_replaced_for_freshness"] += 1
            self.worker_queue_peak = max(self.worker_queue_peak, self.pending_clouds.qsize())

    def _cloud_worker(self) -> None:
        while True:
            pending_item = self.pending_clouds.get()
            try:
                if pending_item is None:
                    return
                message, topic = pending_item
                self._process_cloud_message(message, topic)
            except Exception as exc:  # 后台线程不能因单条坏消息退出，否则 queue.join 会永久等待。
                with self.rows_lock:
                    self.cloud_drop_reasons[f"worker_exception:{type(exc).__name__}:{exc}"] += 1
            finally:
                if pending_item is not None:
                    self.pending_clouds.task_done()

    def _process_cloud_message(self, message: PointCloud2, topic: str) -> None:
        if (not self.defer_sync and self.get_clock().now().nanoseconds - stamp_ns(message.header.stamp) > 750_000_000):
            self.cloud_drop_reasons["OLD_MEASUREMENT_SKIPPED"] += 1
            return
        started = time.perf_counter()
        if self.level_height_filter:
            with self.height_lock:
                if self.height_latest_stamp is None or stamp_ns(message.header.stamp) > self.height_latest_stamp:
                    self.pending_height_clouds.append((message,topic))
                    while len(self.pending_height_clouds) > 100:
                        self.pending_height_clouds.popleft()
                        self.cloud_drop_reasons["HEIGHT_POSE_WAIT_OVERFLOW"] += 1
                    return
        try:
            row = self._make_cloud_row(message, topic)
        except (TypeError, ValueError, AssertionError) as exc:
            try:
                stamp = stamp_ns(message.header.stamp)
            except (TypeError, ValueError):
                stamp = 0
            with self.rows_lock:
                self.cloud_drop_reasons[str(exc)] += 1
            row = CloudRow(self.next_row_id, topic, stamp, str(message.header.frame_id).strip(), (), 0, 0, str(exc))
        with self.rows_lock:
            row = replace(row, row_id=self.next_row_id)
            self.next_row_id += 1
        if row.stamp_ns <= 0:
            return
        with self.rows_lock:
            if row.stamp_ns <= self.last_emitted_group_stamp_ns:
                self.sync_drop_reasons["late_row_after_monotonic_output"] += 1
                return
            queue = self.rows[topic]
            queue.append(row)
            while len(queue) > MAX_QUEUE_ROWS and not self.defer_sync:
                queue.popleft()
                self.sync_drop_reasons["bounded_queue_overflow"] += 1
            if not self.defer_sync:
                clustering_started = time.perf_counter()
                self._drain_groups()
                self.last_group_processing_s = time.perf_counter()-clustering_started
        self.cloud_processing_duration_s = time.perf_counter() - started

    def finish_workers(self) -> None:
        """停止前排空各路点云队列，确保 receipt 反映全部已接收消息。"""

        if self.workers_finished:
            return
        self.pending_clouds.join()
        self.worker_stop.set()
        self.pending_clouds.close()
        self.worker_thread.join(timeout=5.0)
        if self.defer_sync:
            with self.rows_lock:
                self._drain_groups(force=True)
        self.workers_finished = True

    def _drain_groups(self, *, force: bool = False) -> None:
        while True:
            ready = pop_ready_group(
                self.rows,
                tolerance_ns=self.sync_tolerance_ns,
                last_emitted_stamp_ns=self.last_emitted_group_stamp_ns,
                force=force,
            )
            if ready is None:
                return
            rows, primary, stale = ready
            if stale:
                self.sync_drop_reasons["stale_or_late_row_evicted"] += stale
            self.last_emitted_group_stamp_ns = int(primary)
            self.group_sensor_count_histogram[len(rows)] += 1
            if len(rows) < len(POINTCLOUD_TOPICS):
                self.sync_drop_reasons["partial_group_emitted"] += 1
            self._emit_group(rows, primary)

    def _emit_group(self, rows: tuple[CloudRow, ...], primary_stamp: int) -> None:
        self.groups_formed += 1
        groups = [tuple(point for row in rows for point in row.points)] if len({row.stamp_ns for row in rows}) == 1 else [row.points for row in rows]
        candidates_list: list[CandidateObservation] = []
        for points in groups:
            common = dict(
                minimum_support=self.minimum_support,
                maximum_clusters=self.maximum_clusters,
                maximum_vertical_range_delta_m=self.maximum_vertical_range_delta_m,
                maximum_vertical_surface_normal_up_abs=self.maximum_vertical_surface_normal_up_abs,
                minimum_vertical_consistent_neighbors=self.minimum_vertical_consistent_neighbors,
                center_mode=self.cluster_center_mode,
                center_radial_offset_m=self.cluster_center_radial_offset_m,
                funnel=self.funnel,
            )
            if self.organized_dual_path:
                structural = cluster_base_pointcloud(
                    points,
                    voxel_size_m=self.structural_voxel_size_m,
                    neighbor_radius_m=self.structural_cluster_radius_m,
                    organized_structure_only=True,
                    entity_evidence=True,
                    **common,
                )
                complete = cluster_base_pointcloud(
                    points,
                    voxel_size_m=self.voxel_size_m,
                    neighbor_radius_m=self.cluster_radius_m,
                    organized_structure_only=False,
                    entity_evidence=False,
                    **common,
                )
                # 结构通道负责提供实体证据，完整通道负责提供连续位置量测。
                # 证据覆盖到最近的完整候选，避免两条通道为同一物体建立两个轨迹。
                # 一条结构证据只能确认一个完整候选。原先逐个找最近点会让同一
                # 结构簇给周围多个小簇同时盖章，直接放大水面误检。
                pairs = sorted(
                    (
                        (
                            math.hypot(
                                complete_candidate.x_m - seed.x_m,
                                complete_candidate.y_m - seed.y_m,
                            ),
                            complete_index,
                            structural_index,
                        )
                        for complete_index, complete_candidate in enumerate(complete)
                        for structural_index, seed in enumerate(structural)
                    ),
                    key=lambda item: item[0],
                )
                matched_complete: set[int] = set()
                matched_structural: set[int] = set()
                for distance, complete_index, structural_index in pairs:
                    if distance > self.dual_path_evidence_merge_distance_m:
                        break
                    if (
                        complete_index in matched_complete
                        or structural_index in matched_structural
                    ):
                        continue
                    matched_complete.add(complete_index)
                    matched_structural.add(structural_index)
                stamped_complete = [
                    replace(candidate, entity_evidence=True)
                    if index in matched_complete else candidate
                    for index, candidate in enumerate(complete)
                ]
                candidates_list.extend(stamped_complete)
                candidates_list.extend(
                    seed for index, seed in enumerate(structural)
                    if index not in matched_structural
                )
            else:
                candidates_list.extend(cluster_base_pointcloud(
                    points,
                    voxel_size_m=self.voxel_size_m,
                    neighbor_radius_m=self.cluster_radius_m,
                    organized_structure_only=self.organized_structure_only,
                    entity_evidence=self.organized_structure_only,
                    **common,
                ))
        candidates = tuple(candidates_list)
        if not candidates:
            output_candidates: tuple[CandidateObservation, ...] = ()
            self.base_drop_reasons["NO_POINTCLOUD_CANDIDATE_AFTER_SYNC"] += 1
        elif len({row.stamp_ns for row in rows}) == 1:
            output_candidates = fuse_synchronized_candidates(candidates)
        else:
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
            detection.center.z = float(candidate.z_m)
            detection.radius_m = float(candidate.radius_m)
            detection.support = int(candidate.support)
            detection.position_std_m = float(candidate.position_std_m)
            detection.range_m = float(candidate.range_m)
            detection.confidence = float(candidate.confidence)
            detection.source_sensors = list(candidate.source_sensors)
            detection.z_min_m = float(candidate.z_min_m)
            detection.z_max_m = float(candidate.z_max_m)
            detection.vertical_extent_m = float(candidate.vertical_extent_m)
            detection.horizontal_major_extent_m = float(candidate.horizontal_major_extent_m)
            detection.horizontal_minor_extent_m = float(candidate.horizontal_minor_extent_m)
            detection.surface_normal_up_abs = float(candidate.surface_normal_up_abs)
            detection.vertical_voxel_layers = int(candidate.vertical_voxel_layers)
            detection.vertical_surface_fraction = float(candidate.vertical_surface_fraction)
            detection.organized_vertical_stack_fraction = float(candidate.organized_vertical_stack_fraction)
            detection.mean_min_vertical_range_delta_m = float(candidate.mean_min_vertical_range_delta_m)
            detection.organized_normal_valid_fraction = float(candidate.organized_normal_valid_fraction)
            detection.entity_evidence = bool(candidate.entity_evidence)
            output.detections.append(detection)
        self.detections_publisher.publish(output)
        self.base_detection_messages += 1
        self.base_detection_objects += len(output_candidates)
        if output_candidates:
            self.base_detection_nonempty_messages += 1
        else:
            self.base_detection_empty_messages += 1

    def build_receipt(self, status: str) -> dict[str, Any]:
        return {
            "schema": "phase16z_s3_lidar_pointcloud_candidate_runtime_v1",
            "status": status,
            "base_frame": BASE_FRAME,
            "detections_topic": DETECTIONS_TOPIC,
            "pointcloud_topics": list(POINTCLOUD_TOPICS),
            "input_funnel": {
                "five_pointcloud_messages": {topic: int(self.cloud_counts[topic]) for topic in POINTCLOUD_TOPICS},
                "five_pointcloud_messages_total": int(sum(self.cloud_counts.values())),
                "five_pointcloud_raw_points": {topic: int(self.cloud_point_counts[topic]) for topic in POINTCLOUD_TOPICS},
                "five_pointcloud_finite_points": {topic: int(self.cloud_finite_counts[topic]) for topic in POINTCLOUD_TOPICS},
                "sync_groups": int(self.groups_formed),
                "sync_group_width": len(POINTCLOUD_TOPICS),
                "sync_group_sensor_count_histogram": {
                    str(key): int(value)
                    for key, value in sorted(self.group_sensor_count_histogram.items())
                },
                "base_detection_objects": int(self.base_detection_objects),
                "base_detection_messages": int(self.base_detection_messages),
                "base_detection_nonempty_messages": int(self.base_detection_nonempty_messages),
                "base_detection_empty_messages": int(self.base_detection_empty_messages),
                "drop_reasons": {
                    "pointcloud": dict(sorted(self.cloud_drop_reasons.items())),
                    "sync": dict(sorted(self.sync_drop_reasons.items())),
                    "base_detection": dict(sorted(self.base_drop_reasons.items())),
                },
            },
            "pointcloud_funnel": self.funnel.as_dict(),
            "pointcloud_funnel_by_topic": {
                topic: self.cloud_funnels[topic].as_dict() for topic in POINTCLOUD_TOPICS
            },
            "pointcloud_frames": {topic: sorted(self.cloud_frames[topic]) for topic in POINTCLOUD_TOPICS},
            "pointcloud_shape": self.cloud_shape,
            "sensor_tf": {
                "attempts": {topic: int(self.tf_attempts[topic]) for topic in POINTCLOUD_TOPICS},
                "exact_successes": {topic: int(self.tf_exact_successes[topic]) for topic in POINTCLOUD_TOPICS},
                "exact_lookup_misses_recovered": {topic: int(self.tf_exact_misses[topic]) for topic in POINTCLOUD_TOPICS},
                "latest_rigid_fallbacks": {topic: int(self.tf_latest_fallbacks[topic]) for topic in POINTCLOUD_TOPICS},
                "failures": {topic: int(self.tf_failures[topic]) for topic in POINTCLOUD_TOPICS},
            },
            "clock": {
                "message_count": int(self.clock_count),
                "first_stamp_ns": self.clock_first_ns,
                "last_stamp_ns": self.clock_last_ns,
                "regressions": int(self.clock_regressions),
            },
            "cluster_parameters": {
                "voxel_size_m": self.voxel_size_m,
                "cluster_radius_m": self.cluster_radius_m,
                "structural_voxel_size_m": self.structural_voxel_size_m,
                "structural_cluster_radius_m": self.structural_cluster_radius_m,
                "minimum_support": self.minimum_support,
                "maximum_clusters": self.maximum_clusters,
                "ownship_z_min_m": self.ownship_z_min_m,
                "ownship_z_max_m": self.ownship_z_max_m,
                "detection_z_min_m": self.detection_z_min_m,
                "detection_z_max_m": self.detection_z_max_m,
                "maximum_detection_range_m": self.maximum_detection_range_m,
                "sync_tolerance_ms": self.sync_tolerance_ns / 1e6,
                "defer_sync": self.defer_sync,
                "organized_structure_only": self.organized_structure_only,
                "organized_dual_path": self.organized_dual_path,
                "cloud_water_plane_level_filter": self.cloud_water_plane_level_filter,
                "dual_path_evidence_merge_distance_m": self.dual_path_evidence_merge_distance_m,
                "cluster_center_mode": self.cluster_center_mode,
                "cluster_center_radial_offset_m": self.cluster_center_radial_offset_m,
                "maximum_vertical_range_delta_m": self.maximum_vertical_range_delta_m,
                "maximum_vertical_surface_normal_up_abs": self.maximum_vertical_surface_normal_up_abs,
                "minimum_vertical_consistent_neighbors": self.minimum_vertical_consistent_neighbors,
                "worker_threads": 1,
                "worker_queue_capacity": len(POINTCLOUD_TOPICS),
                "worker_replaced_by_topic": dict(self.pending_clouds.replaced_by_topic),
                "last_cloud_processing_s": getattr(self, "cloud_processing_duration_s", None),
                "last_cloud_phases": getattr(self, "last_cloud_phases", None),
                "last_group_processing_s": getattr(self, "last_group_processing_s", None),
                "worker_queue_peak": int(self.worker_queue_peak),
            },
            "fixed_sensor_tf_proof": self.tf_proof,
            "forbidden_topics_subscribed": [],
            "gt_input_subscription_count": 0,
            "world_input_subscription_count": 0,
            "actuator_publishers": 0,
            "elapsed_wall_s": time.monotonic() - self.started_monotonic,
        }

    def publish_status(self) -> None:
        message = String()
        message.data = json.dumps(self.build_receipt("RUNNING"), ensure_ascii=False, sort_keys=True)
        self.status_publisher.publish(message)

    def write_receipt(self) -> None:
        if not self.receipt_path:
            return
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_path.write_text(json.dumps(self.build_receipt("MEASURED"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node: S3LidarPointCloudCandidateNode | None = None
    executor: SingleThreadedExecutor | None = None
    stop_requested = False
    try:
        node = S3LidarPointCloudCandidateNode()
        # 回调只登记消息和姿态；重计算已有独立线程，避免线程池调度争用 GIL。
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        # 保持 rclpy context 有效到后台队列排空后，才能在 finally 中发布最后一批离线检测。
        def request_executor_shutdown(signum: int, frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True
            executor.wake()

        signal.signal(signal.SIGINT, request_executor_shutdown)
        signal.signal(signal.SIGTERM, request_executor_shutdown)
        while not stop_requested and rclpy.ok():
            executor.spin_once(timeout_sec=1.0)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        if node is not None:
            node.finish_workers()
            node.write_receipt()
            node.destroy_node()
        if executor is not None:
            executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
