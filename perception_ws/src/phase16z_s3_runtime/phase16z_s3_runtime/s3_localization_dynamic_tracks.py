"""独立动态跟踪进程，按生产里程计插值观测姿态并发布绝对速度。"""

from __future__ import annotations
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from phase16z_perception_msgs.msg import DynamicDetectionArray, DynamicTrack, DynamicTrackArray
from .s3_async_fusion import fuse_common_frame_candidates
from .s3_candidate_repair import CandidateObservation
from .s3_tracker_repair import RepairedTrackEstimator, TrackEstimate
from .s3_track_semantics import (
    CausalFleetSpeedCalibrator, CausalStaticBackgroundMap, CausalVelocityRefiner,
    SemanticForestModel, SemanticObjectMemory, TrackEvidenceHistory, motion_supported_semantic_probability,
)
from .s3_timestamped_frames import PlanarPose, TimestampedPoseBuffer
from .s3_localization_math import quaternion_to_yaw
from .s3_runtime_contracts import (
    BASE_DETECTIONS_TOPIC, PUBLIC_FRAME, BASE_FRAME, ODOMETRY_TOPIC, TRACKS_TOPIC,
    TRACKING_STATUS_TOPIC, ASYNC_FUSION_MAX_SPAN_NS, DYNAMIC_SPEED_BOUND_MPS, DYNAMIC_SPEED_MIN_MPS,
    BasePoseState, quaternion_from_yaw, stamp_ns, stamp_to_msg, timed_localization_callback,
)


class S3LocalizationDynamicTracks(Node):
    """仅接收候选与生产里程计；导航在另一个进程连续运行。"""
    def __init__(self) -> None:
        super().__init__("phase16z_s3_localization_dynamic_tracks")
        self.declare_parameter("receipt_path", "")
        self.declare_parameter("base_detections_topic", BASE_DETECTIONS_TOPIC)
        self.declare_parameter("dynamic_publish_speed_gate_mps", 0.0)
        self.declare_parameter("tracker_confirmation_hits", 3)
        self.declare_parameter("tracker_dynamic_persistence_s", 0.30)
        self.declare_parameter("tracker_maximum_track_age_s", 4.0)
        self.declare_parameter("minimum_candidate_enu_z_m", 0.11)
        self.declare_parameter("minimum_public_hit_count", 9)
        self.declare_parameter("tracker_measurement_floor_m", 0.15)
        self.declare_parameter("tracker_partial_surface_std_m", 0.65)
        self.declare_parameter("tracker_minimum_entity_evidence_hits", 0)
        self.declare_parameter("tracker_require_entity_evidence_for_spawn", False)
        self.declare_parameter("semantic_filter_enabled", False)
        self.declare_parameter("semantic_model_path", "")
        self.declare_parameter("semantic_primary_minimum_hit_count", 5)
        self.declare_parameter("semantic_birth_probability", 0.90)
        self.declare_parameter("semantic_association_position_gate_m", 2.0)
        self.declare_parameter("semantic_association_velocity_gate_mps", 0.7)
        self.declare_parameter("semantic_maximum_memory_age_s", 5.0)
        self.declare_parameter("semantic_publication_coast_s", 0.7)
        self.declare_parameter("semantic_spawn_separation_m", 2.0)
        self.declare_parameter("semantic_velocity_previous_weight", 0.5)
        self.declare_parameter("semantic_minimum_birth_evidence_hits", 3)
        self.declare_parameter("semantic_birth_evidence_max_gap_s", 0.35)
        self.declare_parameter("semantic_maximum_background_frames", 20)
        self.declare_parameter("semantic_background_cell_size_m", 0.5)
        self.declare_parameter("semantic_background_query_radius_m", 0.8)
        self.declare_parameter("semantic_background_history_window_s", 12.0)
        self.declare_parameter("semantic_background_guard_s", 1.5)
        self.declare_parameter("semantic_velocity_refinement_enabled", True)
        self.declare_parameter("semantic_velocity_history_window_s", 15.0)
        self.declare_parameter("semantic_velocity_minimum_span_s", 3.0)
        self.declare_parameter("semantic_velocity_circle_direction_weight", 0.4)
        self.declare_parameter("semantic_velocity_history_speed_weight", 0.55)
        self.declare_parameter("semantic_velocity_phase_lead_s", -0.1)
        self.declare_parameter("semantic_velocity_fallback_heading_window_s", 2.0)
        self.declare_parameter("semantic_velocity_fallback_minimum_span_s", 1.0)
        self.declare_parameter("semantic_velocity_fallback_phase_lead_s", 1.0)
        self.declare_parameter("semantic_velocity_fallback_minimum_samples", 4)
        self.declare_parameter("semantic_fleet_speed_calibration_enabled", False)
        self.declare_parameter("semantic_fleet_speed_history_window_s", 10.0)
        self.declare_parameter("semantic_fleet_speed_prior_weight", 0.30)
        receipt_text = str(self.get_parameter("receipt_path").value).strip()
        self.receipt_path = Path(receipt_text) if receipt_text else None
        self.base_detections_topic = str(self.get_parameter("base_detections_topic").value).strip()
        self.dynamic_publish_speed_gate_mps = float(
            self.get_parameter("dynamic_publish_speed_gate_mps").value
        )
        self.tracker_confirmation_hits = int(self.get_parameter("tracker_confirmation_hits").value)
        self.tracker_dynamic_persistence_s = float(self.get_parameter("tracker_dynamic_persistence_s").value)
        self.tracker_maximum_track_age_s = float(
            self.get_parameter("tracker_maximum_track_age_s").value
        )
        self.minimum_candidate_enu_z_m = float(self.get_parameter("minimum_candidate_enu_z_m").value)
        self.minimum_public_hit_count = int(self.get_parameter("minimum_public_hit_count").value)
        self.tracker_measurement_floor_m = float(self.get_parameter("tracker_measurement_floor_m").value)
        self.tracker_partial_surface_std_m = float(self.get_parameter("tracker_partial_surface_std_m").value)
        self.tracker_minimum_entity_evidence_hits = int(
            self.get_parameter("tracker_minimum_entity_evidence_hits").value
        )
        self.tracker_require_entity_evidence_for_spawn = bool(
            self.get_parameter("tracker_require_entity_evidence_for_spawn").value
        )
        self.semantic_filter_enabled = bool(self.get_parameter("semantic_filter_enabled").value)
        self.semantic_model_path = str(self.get_parameter("semantic_model_path").value).strip()
        self.semantic_primary_minimum_hit_count = int(
            self.get_parameter("semantic_primary_minimum_hit_count").value
        )
        self.semantic_birth_probability = float(
            self.get_parameter("semantic_birth_probability").value
        )
        self.semantic_association_position_gate_m = float(
            self.get_parameter("semantic_association_position_gate_m").value
        )
        self.semantic_association_velocity_gate_mps = float(
            self.get_parameter("semantic_association_velocity_gate_mps").value
        )
        self.semantic_maximum_memory_age_s = float(
            self.get_parameter("semantic_maximum_memory_age_s").value
        )
        self.semantic_publication_coast_s = float(
            self.get_parameter("semantic_publication_coast_s").value
        )
        self.semantic_spawn_separation_m = float(
            self.get_parameter("semantic_spawn_separation_m").value
        )
        self.semantic_velocity_previous_weight = float(
            self.get_parameter("semantic_velocity_previous_weight").value
        )
        self.semantic_minimum_birth_evidence_hits = int(
            self.get_parameter("semantic_minimum_birth_evidence_hits").value
        )
        self.semantic_birth_evidence_max_gap_s = float(
            self.get_parameter("semantic_birth_evidence_max_gap_s").value
        )
        self.semantic_maximum_background_frames = int(
            self.get_parameter("semantic_maximum_background_frames").value
        )
        self.semantic_background_cell_size_m = float(
            self.get_parameter("semantic_background_cell_size_m").value
        )
        self.semantic_background_query_radius_m = float(
            self.get_parameter("semantic_background_query_radius_m").value
        )
        self.semantic_background_history_window_s = float(
            self.get_parameter("semantic_background_history_window_s").value
        )
        self.semantic_background_guard_s = float(
            self.get_parameter("semantic_background_guard_s").value
        )
        self.semantic_velocity_refinement_enabled = bool(
            self.get_parameter("semantic_velocity_refinement_enabled").value
        )
        self.semantic_velocity_history_window_s = float(
            self.get_parameter("semantic_velocity_history_window_s").value
        )
        self.semantic_velocity_minimum_span_s = float(
            self.get_parameter("semantic_velocity_minimum_span_s").value
        )
        self.semantic_velocity_circle_direction_weight = float(
            self.get_parameter("semantic_velocity_circle_direction_weight").value
        )
        self.semantic_velocity_history_speed_weight = float(
            self.get_parameter("semantic_velocity_history_speed_weight").value
        )
        self.semantic_velocity_phase_lead_s = float(
            self.get_parameter("semantic_velocity_phase_lead_s").value
        )
        self.semantic_velocity_fallback_heading_window_s = float(
            self.get_parameter("semantic_velocity_fallback_heading_window_s").value
        )
        self.semantic_velocity_fallback_minimum_span_s = float(
            self.get_parameter("semantic_velocity_fallback_minimum_span_s").value
        )
        self.semantic_velocity_fallback_phase_lead_s = float(
            self.get_parameter("semantic_velocity_fallback_phase_lead_s").value
        )
        self.semantic_velocity_fallback_minimum_samples = int(
            self.get_parameter("semantic_velocity_fallback_minimum_samples").value
        )
        self.semantic_fleet_speed_calibration_enabled = bool(
            self.get_parameter("semantic_fleet_speed_calibration_enabled").value
        )
        self.semantic_fleet_speed_history_window_s = float(
            self.get_parameter("semantic_fleet_speed_history_window_s").value
        )
        self.semantic_fleet_speed_prior_weight = float(
            self.get_parameter("semantic_fleet_speed_prior_weight").value
        )
        if not math.isfinite(self.dynamic_publish_speed_gate_mps) or self.dynamic_publish_speed_gate_mps < 0.0:
            raise RuntimeError("DYNAMIC_PUBLISH_SPEED_GATE_INVALID")
        if self.dynamic_publish_speed_gate_mps > 0.0:
            raise RuntimeError("DYNAMIC_SPEED_MAGNITUDE_GATE_FORBIDDEN")
        if (
            self.tracker_confirmation_hits < 1
            or not math.isfinite(self.tracker_dynamic_persistence_s)
            or self.tracker_dynamic_persistence_s < 0.0
            or not math.isfinite(self.tracker_maximum_track_age_s)
            or self.tracker_maximum_track_age_s <= 0.0
        ):
            raise RuntimeError("TRACKER_DYNAMIC_CONFIRMATION_PARAMETER_INVALID")
        if (
            not math.isfinite(self.minimum_candidate_enu_z_m)
            or self.minimum_public_hit_count < self.tracker_confirmation_hits
            or not math.isfinite(self.tracker_measurement_floor_m)
            or not math.isfinite(self.tracker_partial_surface_std_m)
            or self.tracker_measurement_floor_m <= 0.0
            or self.tracker_partial_surface_std_m < 0.0
            or self.tracker_minimum_entity_evidence_hits < 0
        ):
            raise RuntimeError("TRACKER_PUBLICATION_OR_MEASUREMENT_PARAMETER_INVALID")
        if (
            self.semantic_primary_minimum_hit_count < self.tracker_confirmation_hits
            or not (0.0 <= self.semantic_birth_probability <= 1.0)
            or min(
                self.semantic_association_position_gate_m,
                self.semantic_association_velocity_gate_mps,
                self.semantic_maximum_memory_age_s,
                self.semantic_publication_coast_s,
                self.semantic_spawn_separation_m,
            ) <= 0.0
            or not (0.0 <= self.semantic_velocity_previous_weight <= 1.0)
            or self.semantic_minimum_birth_evidence_hits < 1
            or self.semantic_birth_evidence_max_gap_s <= 0.0
            or self.semantic_maximum_background_frames < 0
            or min(
                self.semantic_background_cell_size_m,
                self.semantic_background_query_radius_m,
                self.semantic_background_history_window_s,
                self.semantic_background_guard_s,
            ) <= 0.0
            or self.semantic_background_guard_s >= self.semantic_background_history_window_s
            or min(
                self.semantic_velocity_history_window_s,
                self.semantic_velocity_minimum_span_s,
                self.semantic_velocity_fallback_heading_window_s,
                self.semantic_velocity_fallback_minimum_span_s,
                self.semantic_fleet_speed_history_window_s,
            ) <= 0.0
            or self.semantic_velocity_minimum_span_s > self.semantic_velocity_history_window_s
            or self.semantic_velocity_fallback_minimum_span_s > self.semantic_velocity_fallback_heading_window_s
            or not 0.0 <= self.semantic_velocity_circle_direction_weight <= 1.0
            or not 0.0 <= self.semantic_velocity_history_speed_weight <= 1.0
            or self.semantic_velocity_fallback_minimum_samples < 2
            or not 0.0 <= self.semantic_fleet_speed_prior_weight <= 1.0
        ):
            raise RuntimeError("SEMANTIC_FILTER_PARAMETER_INVALID")
        if self.semantic_filter_enabled and (
            not self.semantic_model_path or self.tracker_require_entity_evidence_for_spawn
        ):
            raise RuntimeError("SEMANTIC_FILTER_MODEL_OR_SPAWN_CONTRACT_INVALID")
        output_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.status_publisher = self.create_publisher(String, TRACKING_STATUS_TOPIC, output_qos)
        self.tracks_publisher = self.create_publisher(DynamicTrackArray, TRACKS_TOPIC, output_qos)
        self.base_detections_subscription = self.create_subscription(
            DynamicDetectionArray,
            self.base_detections_topic,
            self.base_detections_callback,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.started_monotonic = time.monotonic()
        self.base_detection_message_count = 0
        self.base_detection_pose_count = 0
        self.base_detection_rejected_count = 0
        self.base_detection_enu_attempt_count = 0
        self.base_detection_enu_success_count = 0
        self.base_detection_enu_failure_count = 0
        self.base_detection_drop_reasons: dict[str, int] = defaultdict(int)
        self.association_update_count = 0
        self.association_input_detection_count = 0
        self.association_match_count = 0
        self.association_unmatched_detection_count = 0
        self.association_unmatched_track_count = 0
        self.tracker_tentative_observation_count = 0
        self.tracker_confirmed_observation_count = 0
        self.tracker_tentative_peak = 0
        self.tracker_confirmed_peak = 0
        self.tracker_spawn_count = 0
        self.tracker_deleted_count = 0
        self.dynamic_filter_drop_reasons: dict[str, int] = defaultdict(int)
        self.semantic_scored_track_count = 0
        self.semantic_high_probability_track_count = 0
        self.semantic_motion_supported_track_count = 0
        self.semantic_history_count_peak = 0
        self.pending_base_detections = deque()
        self.velocity_valid_output_count = 0
        self.velocity_measured_output_count = 0
        self.velocity_coasting_output_count = 0
        self.velocity_invalid_output_count = 0
        self.dynamic_tracks_output_count = 0
        self.dynamic_track_message_count = 0
        self.dynamic_track_count = 0
        self.dynamic_empty_output_count = 0
        self.pose_buffer = TimestampedPoseBuffer(maximum_gap_ns=250_000_000)
        self.pose_interpolation_attempt_count = 0
        self.pose_interpolation_success_count = 0
        self.pose_interpolation_failure_count = 0
        self.pose_interpolation_drop_reasons: dict[str, int] = defaultdict(int)
        self.last_localization_stamp_ns: int | None = None
        self._last_localization_wall = None
        self.last_tracks_stamp_ns: int | None = None
        self.last_status_reason = "WAITING_FOR_PRODUCTION_ODOMETRY"
        self.semantic_histories: dict[int, TrackEvidenceHistory] = defaultdict(TrackEvidenceHistory)
        self.last_semantic_stamp_ns: int | None = None
        self.semantic_model = (
            SemanticForestModel(self.semantic_model_path)
            if self.semantic_filter_enabled
            else None
        )
        self.semantic_object_memory = (
            SemanticObjectMemory(
                birth_probability=self.semantic_birth_probability,
                association_position_gate_m=self.semantic_association_position_gate_m,
                association_velocity_gate_mps=self.semantic_association_velocity_gate_mps,
                maximum_memory_age_s=self.semantic_maximum_memory_age_s,
                publication_coast_s=self.semantic_publication_coast_s,
                spawn_separation_m=self.semantic_spawn_separation_m,
                velocity_previous_weight=self.semantic_velocity_previous_weight,
                minimum_birth_evidence_hits=self.semantic_minimum_birth_evidence_hits,
                birth_evidence_max_gap_s=self.semantic_birth_evidence_max_gap_s,
            )
            if self.semantic_filter_enabled
            else None
        )
        self.semantic_background_map = (
            CausalStaticBackgroundMap(
                cell_size_m=self.semantic_background_cell_size_m,
                query_radius_m=self.semantic_background_query_radius_m,
                history_window_s=self.semantic_background_history_window_s,
                guard_s=self.semantic_background_guard_s,
            )
            if self.semantic_filter_enabled
            else None
        )
        self.semantic_velocity_refiner = (
            CausalVelocityRefiner(
                history_window_s=self.semantic_velocity_history_window_s,
                minimum_span_s=self.semantic_velocity_minimum_span_s,
                circle_direction_weight=self.semantic_velocity_circle_direction_weight,
                history_speed_weight=self.semantic_velocity_history_speed_weight,
                phase_lead_s=self.semantic_velocity_phase_lead_s,
                fallback_heading_window_s=self.semantic_velocity_fallback_heading_window_s,
                fallback_minimum_span_s=self.semantic_velocity_fallback_minimum_span_s,
                fallback_phase_lead_s=self.semantic_velocity_fallback_phase_lead_s,
                fallback_minimum_samples=self.semantic_velocity_fallback_minimum_samples,
            )
            if self.semantic_filter_enabled and self.semantic_velocity_refinement_enabled
            else None
        )
        self.semantic_fleet_speed_calibrator = (
            CausalFleetSpeedCalibrator(
                history_window_s=self.semantic_fleet_speed_history_window_s,
                prior_weight=self.semantic_fleet_speed_prior_weight,
            )
            if self.semantic_filter_enabled and self.semantic_fleet_speed_calibration_enabled
            else None
        )
        self.tracker = RepairedTrackEstimator(
            PUBLIC_FRAME,
            confirmation_hits=self.tracker_confirmation_hits,
            maximum_missed_updates=5,
            maximum_track_age_s=self.tracker_maximum_track_age_s,
            dynamic_speed_threshold_mps=DYNAMIC_SPEED_MIN_MPS,
            dynamic_persistence_s=self.tracker_dynamic_persistence_s,
            measurement_floor_m=self.tracker_measurement_floor_m,
            partial_surface_std_m=self.tracker_partial_surface_std_m,
            minimum_entity_evidence_hits=self.tracker_minimum_entity_evidence_hits,
            require_entity_evidence_for_spawn=self.tracker_require_entity_evidence_for_spawn,
        )

        self.callback_metrics = {}
        self.odometry_message_count = 0
        self.odometry_rejected_count = 0
        self.navigation_subscription = self.create_subscription(
            Odometry, ODOMETRY_TOPIC, self.navigation_callback,
            QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))

    @timed_localization_callback
    def navigation_callback(self, message: Odometry):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        v = message.twist.twist.linear
        try:
            stamp = stamp_ns(message.header.stamp)
        except ValueError:
            self.odometry_rejected_count += 1
            self.last_status_reason = "ODOMETRY_STAMP_INVALID"
            return
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z)
        if (message.header.frame_id != PUBLIC_FRAME or message.child_frame_id != BASE_FRAME
                or not all(math.isfinite(value) for value in values)
                or abs(sum(value * value for value in values[3:7]) - 1.0) > 0.02):
            self.odometry_rejected_count += 1
            self.last_status_reason = "ODOMETRY_FRAME_OR_POSE_INVALID"
            return
        if self.last_localization_stamp_ns is not None and stamp <= self.last_localization_stamp_ns:
            return
        velocity_enu = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply([v.x, v.y, v.z])
        self.pose_buffer.add(PlanarPose(stamp, p.x, p.y, quaternion_to_yaw(q.x, q.y, q.z, q.w),
                                       p.z, (q.x, q.y, q.z, q.w), tuple(velocity_enu[:2])))
        self.pose_buffer.discard_before(stamp - 10_000_000_000)
        self.last_localization_stamp_ns = stamp
        self._last_localization_wall = time.monotonic()
        self.odometry_message_count += 1
        self._drain_pending_detections()

    def _nearest_pose(self, stamp: int) -> BasePoseState | None:
        self.pose_interpolation_attempt_count += 1
        try:
            pose = self.pose_buffer.interpolate(int(stamp))
        except ValueError as exc:
            reason = str(exc)
            self.pose_interpolation_failure_count += 1
            self.pose_interpolation_drop_reasons[reason] += 1
            return None
        self.pose_interpolation_success_count += 1
        return BasePoseState(
            stamp_ns=int(pose.stamp_ns),
            x_m=float(pose.x_enu_m),
            y_m=float(pose.y_enu_m),
            z_m=float(pose.z_m),
            yaw_rad=float(pose.yaw_rad),
            quaternion_xyzw=pose.quaternion_xyzw or quaternion_from_yaw(pose.yaw_rad),
        )

    @timed_localization_callback
    def base_detections_callback(self, message: DynamicDetectionArray) -> None:
        # 等待已经在途的生产里程计端点；等待计入真实输出延迟，不读未来文件。
        self.pending_base_detections.append(message)
        while len(self.pending_base_detections) > 2:
            self.pending_base_detections.popleft()
            self.base_detection_drop_reasons["PENDING_NEWER_MEASUREMENT_REPLACED"] += 1
        self._drain_pending_detections()

    def _drain_pending_detections(self) -> None:
        while self.pending_base_detections:
            message = self.pending_base_detections[0]
            stamp = stamp_ns(message.header.stamp)
            if self.last_localization_stamp_ns is None or stamp > self.last_localization_stamp_ns:
                break
            self.pending_base_detections.popleft()
            self._process_base_detections(message)

    def _semantic_public_estimates(
        self,
        stamp: int,
        detections: tuple[CandidateObservation, ...],
        before_track_ids: list[int],
        next_track_id_before_update: int,
        estimates: tuple[TrackEstimate, ...],
    ) -> tuple[TrackEstimate, ...]:
        """更新底层轨迹证据，再输出跨底层 ID 连续的船级对象。"""
        if (
            self.semantic_model is None
            or self.semantic_object_memory is None
            or self.semantic_background_map is None
        ):
            raise RuntimeError("SEMANTIC_FILTER_NOT_INITIALIZED")
        for track_index, observation_index in self.tracker.last_association.matches:
            self.semantic_histories[before_track_ids[track_index]].add_observation(
                detections[observation_index]
            )
        for offset, observation_index in enumerate(
            self.tracker.last_association.unmatched_detection_indices
        ):
            self.semantic_histories[next_track_id_before_update + offset].add_observation(
                detections[observation_index]
            )
        dt = (
            0.1
            if self.last_semantic_stamp_ns is None
            else max(1e-3, (int(stamp) - self.last_semantic_stamp_ns) / 1e9)
        )
        for estimate in estimates:
            self.semantic_histories[estimate.track_id].add_velocity(
                stamp, estimate.vx, estimate.vy, dt
            )
        self.last_semantic_stamp_ns = int(stamp)

        navigation = self.pose_buffer.interpolate(stamp)
        if navigation.velocity_enu_xy_mps is None:
            raise ValueError("TRACKING_TIMESTAMPED_NAVIGATION_VELOCITY_MISSING")
        state = np.asarray([navigation.x_enu_m, navigation.y_enu_m, *navigation.velocity_enu_xy_mps])
        ownship_xyv = (float(state[0]), float(state[1]), float(state[2]), float(state[3]))
        scored: list[tuple[TrackEstimate, float]] = []
        for estimate in estimates:
            # 底层 BOAT 判定依赖高置信速度下界；稀疏点云下速度协方差较大，
            # 若先要求 BOAT 再进入语义模型会形成循环阻塞。确认且成熟的轨迹
            # 均进入语义层，由外形、因果背景和连续 ENU 运动共同决定是否出生。
            if int(estimate.hit_count) < self.semantic_primary_minimum_hit_count:
                continue
            history = self.semantic_histories.get(estimate.track_id)
            if history is None:
                continue
            feature_row = history.feature_row(stamp, estimate, ownship_xyv)
            background = self.semantic_background_map.features(
                stamp, estimate.x, estimate.y
            )
            feature_row.update(background)
            if int(background["causal_background_frames"]) > self.semantic_maximum_background_frames:
                self.dynamic_filter_drop_reasons["CAUSAL_STATIC_BACKGROUND"] += 1
                continue
            model_probability = self.semantic_model.predict_probability(feature_row)
            probability, motion_supported = motion_supported_semantic_probability(
                feature_row,
                model_probability,
                # 连续、低残差的真实运动已经是独立的船体证据。补强后的概率必须
                # 能越过当前出生门槛，否则把门槛调高后会让该通路逻辑上永远失效。
                supported_probability=self.semantic_birth_probability,
                classification=str(estimate.classification),
            )
            self.semantic_motion_supported_track_count += int(motion_supported)
            if str(estimate.classification) != "BOAT" and not motion_supported:
                self.dynamic_filter_drop_reasons["UNKNOWN_WITHOUT_COHERENT_MOTION"] += 1
                continue
            scored.append((estimate, probability))
            self.semantic_scored_track_count += 1
            self.semantic_high_probability_track_count += int(
                probability >= self.semantic_birth_probability
            )
        active_track_ids = set(self.tracker.tracks)
        for track_id in tuple(self.semantic_histories):
            if track_id not in active_track_ids:
                del self.semantic_histories[track_id]
        self.semantic_history_count_peak = max(
            self.semantic_history_count_peak, len(self.semantic_histories)
        )
        self.semantic_background_map.observe(stamp, detections)
        public = self.semantic_object_memory.update(stamp, tuple(scored))
        if self.semantic_velocity_refiner is not None:
            public = self.semantic_velocity_refiner.update(stamp, public)
        if self.semantic_fleet_speed_calibrator is not None:
            public = self.semantic_fleet_speed_calibrator.update(stamp, public)
        return public

    @timed_localization_callback
    def _process_base_detections(self, message: DynamicDetectionArray) -> None:
        self.base_detection_message_count += 1
        self.base_detection_pose_count += len(message.detections)
        try:
            array_stamp = stamp_ns(message.header.stamp)
            if str(message.header.frame_id).strip() != BASE_FRAME:
                raise ValueError("BASE_DETECTION_FRAME_INVALID")
            common_frame_candidates: list[CandidateObservation] = []
            for item in message.detections:
                self.base_detection_enu_attempt_count += 1
                detection_stamp = stamp_ns(item.header.stamp)
                if str(item.header.frame_id).strip() != BASE_FRAME:
                    self.base_detection_enu_failure_count += 1
                    self.base_detection_drop_reasons["BASE_DETECTION_FRAME_INVALID"] += 1
                    continue
                pose = self._nearest_pose(detection_stamp)
                if pose is None:
                    self.base_detection_enu_failure_count += 1
                    self.base_detection_drop_reasons["BASE_DETECTION_LOCALIZATION_TIME_SYNC_UNAVAILABLE"] += 1
                    continue
                x_local, y_local = float(item.center.x), float(item.center.y)
                if not all(math.isfinite(value) for value in (x_local, y_local, float(item.position_std_m))):
                    self.base_detection_enu_failure_count += 1
                    self.base_detection_drop_reasons["BASE_DETECTION_NONFINITE"] += 1
                    continue
                enu_x, enu_y, enu_z = Rotation.from_quat(pose.quaternion_xyzw).apply(
                    [x_local, y_local, float(item.center.z)]) + np.array([pose.x_m, pose.y_m, pose.z_m])
                if not all(math.isfinite(value) for value in (enu_x, enu_y)):
                    self.base_detection_enu_failure_count += 1
                    self.base_detection_drop_reasons["ENU_TRANSFORM_NONFINITE"] += 1
                    continue
                # ENU 的 z=0 对应 SDF 水面基准；去除贴水反射质心，同时保留
                # 只有一个有效体素的远距离船体回波。
                if float(enu_z) < self.minimum_candidate_enu_z_m:
                    self.base_detection_enu_failure_count += 1
                    self.base_detection_drop_reasons["CANDIDATE_BELOW_WATER_CLEARANCE"] += 1
                    continue
                self.base_detection_enu_success_count += 1
                common_frame_candidates.append(
                    CandidateObservation(
                        stamp_ns=detection_stamp,
                        observation_id=str(item.observation_id) or f"ros:{detection_stamp}:{self.base_detection_enu_attempt_count}",
                        x_m=enu_x,
                        y_m=enu_y,
                        z_m=float(enu_z),
                        radius_m=max(0.20, float(item.radius_m)),
                        support=max(1, int(item.support)),
                        source_sensors=tuple(str(value) for value in item.source_sensors if str(value)) or ("unknown",),
                        position_std_m=max(0.05, float(item.position_std_m)),
                        range_m=max(0.05, float(item.range_m)),
                        confidence=min(0.99, max(0.01, float(item.confidence))),
                        observation_stamps_ns=(detection_stamp,),
                        z_min_m=float(item.z_min_m),
                        z_max_m=float(item.z_max_m),
                        vertical_extent_m=float(item.vertical_extent_m),
                        horizontal_major_extent_m=float(item.horizontal_major_extent_m),
                        horizontal_minor_extent_m=float(item.horizontal_minor_extent_m),
                        surface_normal_up_abs=float(item.surface_normal_up_abs),
                        vertical_voxel_layers=max(1, int(item.vertical_voxel_layers)),
                        vertical_surface_fraction=float(item.vertical_surface_fraction),
                        organized_vertical_stack_fraction=float(item.organized_vertical_stack_fraction),
                        mean_min_vertical_range_delta_m=float(item.mean_min_vertical_range_delta_m),
                        organized_normal_valid_fraction=float(item.organized_normal_valid_fraction),
                        entity_evidence=bool(item.entity_evidence),
                    )
                )
            if common_frame_candidates:
                stamp = max(int(item.stamp_ns) for item in common_frame_candidates)
                detections = fuse_common_frame_candidates(
                    common_frame_candidates,
                    fusion_stamp_ns=stamp,
                    maximum_time_span_ns=ASYNC_FUSION_MAX_SPAN_NS,
                    target_speed_bound_mps=DYNAMIC_SPEED_BOUND_MPS,
                )
            else:
                stamp = array_stamp
                detections = ()
            if self.last_tracks_stamp_ns is not None and stamp <= self.last_tracks_stamp_ns:
                self.base_detection_drop_reasons["TRACK_INPUT_STAMP_NOT_INCREASING"] += 1
                return
            expiry_ns = int(self.tracker.maximum_track_age_s * 1_000_000_000)
            before_track_ids = [
                track_id
                for track_id in sorted(self.tracker.tracks)
                if stamp - int(self.tracker.tracks[track_id].last_measurement_stamp_ns) <= expiry_ns
            ]
            next_track_id_before_update = self.tracker.next_track_id
            estimates = self.tracker.update(stamp, detections, frame_id=PUBLIC_FRAME)
            all_estimates = self.tracker.estimates(stamp, include_tentative=True, include_coasting=True)
            self.association_update_count += 1
            self.association_input_detection_count += len(detections)
            self.association_match_count += len(self.tracker.last_association.matches)
            self.association_unmatched_detection_count += len(self.tracker.last_association.unmatched_detection_indices)
            self.association_unmatched_track_count += len(self.tracker.last_association.unmatched_track_indices)
            tracker_funnel = self.tracker.funnel()
            self.tracker_tentative_observation_count += int(tracker_funnel["tentative_tracks"])
            self.tracker_confirmed_observation_count += int(tracker_funnel["confirmed_tracks"])
            self.tracker_tentative_peak = max(self.tracker_tentative_peak, int(tracker_funnel["tentative_tracks"]))
            self.tracker_confirmed_peak = max(self.tracker_confirmed_peak, int(tracker_funnel["confirmed_tracks"]))
            self.tracker_spawn_count += self.tracker.last_spawn_count
            self.tracker_deleted_count += self.tracker.last_deleted_count
            # 速度大小不是动态船语义门。确认轨迹即使慢速或暂时停船也必须发布；
            # 这里只拒绝非有限或标量与向量不一致的数值。
            if self.semantic_filter_enabled:
                publish_estimates = list(self._semantic_public_estimates(
                    stamp,
                    tuple(detections),
                    before_track_ids,
                    next_track_id_before_update,
                    estimates,
                ))
            else:
                publish_estimates = []
                for estimate in estimates:
                    if str(estimate.classification) != "BOAT":
                        self.dynamic_filter_drop_reasons["STATIC_OR_INSUFFICIENT_DYNAMIC_EVIDENCE"] += 1
                        continue
                    if int(estimate.hit_count) < self.minimum_public_hit_count:
                        self.dynamic_filter_drop_reasons["TRACK_NOT_MATURE_FOR_PUBLICATION"] += 1
                        continue
                    publish_estimates.append(estimate)
            finite_publish_estimates = []
            for estimate in publish_estimates:
                values = (float(estimate.vx), float(estimate.vy), float(estimate.speed_mps))
                scalar_consistent = abs(values[2] - math.hypot(values[0], values[1])) <= 1e-6
                if all(math.isfinite(value) for value in values) and scalar_consistent:
                    finite_publish_estimates.append(estimate)
                else:
                    self.dynamic_filter_drop_reasons["VELOCITY_NONFINITE_OR_SCALAR_MISMATCH"] += 1
                    self.velocity_invalid_output_count += 1
            if not all_estimates and not detections:
                self.base_detection_drop_reasons["NO_OBJECTS_REACHED_TRACKER"] += 1
            self._publish_tracks(stamp, tuple(finite_publish_estimates))
            self.dynamic_track_message_count += 1
        except (TypeError, ValueError) as exc:
            self.base_detection_rejected_count += 1
            reason = str(exc)
            self.base_detection_drop_reasons[reason] += 1
            self.last_status_reason = reason

    def _publish_tracks(self, stamp: int, estimates: tuple[Any, ...]) -> None:
        valid_hit_threshold = (
            self.semantic_primary_minimum_hit_count
            if self.semantic_filter_enabled
            else self.minimum_public_hit_count
        )
        array = DynamicTrackArray()
        array.header.stamp = stamp_to_msg(stamp)
        array.header.frame_id = PUBLIC_FRAME
        for estimate in estimates:
            item = DynamicTrack()
            item.header = array.header
            item.track_id = int(estimate.track_id)
            item.pose.position.x = float(estimate.x)
            item.pose.position.y = float(estimate.y)
            item.pose.orientation.w = 1.0
            item.velocity.linear.x = float(estimate.vx)
            item.velocity.linear.y = float(estimate.vy)
            item.speed_mps = float(estimate.speed_mps)
            item.heading_rad = float(estimate.heading_rad)
            item.confidence = float(estimate.confidence)
            covariance = estimate.covariance
            item.position_covariance_xx = float(covariance[0])
            item.position_covariance_xy = float(covariance[1])
            item.position_covariance_yy = float(covariance[5])
            item.velocity_covariance_xx = float(covariance[10])
            item.velocity_covariance_xy = float(covariance[11])
            item.velocity_covariance_yy = float(covariance[15])
            item.age_sec = float(estimate.age_sec)
            item.hit_count = int(estimate.hit_count)
            item.missed_count = int(estimate.missed_count)
            # UNKNOWN=0、BOAT=1 是公共消息合同值。
            item.classification = 1 if str(estimate.classification) == "BOAT" else 0
            array.tracks.append(item)
            self.velocity_valid_output_count += int(
                all(math.isfinite(value) for value in (float(estimate.vx), float(estimate.vy), float(estimate.speed_mps)))
                and estimate.hit_count >= valid_hit_threshold
            )
            if estimate.missed_count == 0:
                self.velocity_measured_output_count += 1
            else:
                self.velocity_coasting_output_count += 1
        self.tracks_publisher.publish(array)
        self.last_status_reason = "TRACKING_OUTPUT_READY"
        self.dynamic_tracks_output_count += 1
        self.dynamic_track_count += len(estimates)
        if not estimates:
            self.dynamic_empty_output_count += 1
        self.last_tracks_stamp_ns = stamp

    def publish_status(self):
        now = self.get_clock().now().nanoseconds
        age = None if self.last_tracks_stamp_ns is None else (now - self.last_tracks_stamp_ns) / 1e9
        payload = {
            "schema": "phase16z_dynamic_tracking_status_v2", "status": "READY" if age is not None and 0 <= age <= 2.0 else "NOT_READY",
            "reason": self.last_status_reason, "callback_metrics": self.callback_metrics,
            "odometry_message_count": self.odometry_message_count, "odometry_rejected_count": self.odometry_rejected_count,
            "tracks_measurement_age_s": age, "dynamic_tracks_output_count": self.dynamic_tracks_output_count,
            "velocity_valid_output_count": self.velocity_valid_output_count,
        }
        self.status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def build_receipt(self):
        return {
            "schema": "phase16z_dynamic_tracking_receipt_v2",
            "runtime_status": "MEASURED" if self.dynamic_tracks_output_count else "NOT_MEASURABLE",
            "public_frame": PUBLIC_FRAME, "base_frame": BASE_FRAME,
            "input_topics": [ODOMETRY_TOPIC, self.base_detections_topic],
            "odometry_messages": self.odometry_message_count, "callback_metrics": self.callback_metrics,
            "pose_interpolation_attempts": self.pose_interpolation_attempt_count,
            "pose_interpolation_successes": self.pose_interpolation_success_count,
            "pose_interpolation_failures": self.pose_interpolation_failure_count,
            "pose_interpolation_drop_reasons": dict(self.pose_interpolation_drop_reasons),
            'gt_input_subscription_count': 0,
            'dynamic_tracks': {
                "topic": TRACKS_TOPIC,
                "type": "phase16z_perception_msgs/msg/DynamicTrackArray",
                "messages": self.dynamic_tracks_output_count,
                "tracks_total": self.dynamic_track_count,
                "empty_messages": self.dynamic_empty_output_count,
                "input_base_messages": self.base_detection_message_count,
                "input_base_poses": self.base_detection_pose_count,
                "rejected_base_messages": self.base_detection_rejected_count,
                "funnel": {
                    "base_detection_objects": self.base_detection_pose_count,
                    "enu_transform_attempts": self.base_detection_enu_attempt_count,
                    "enu_transform_successes": self.base_detection_enu_success_count,
                    "enu_transform_failures": self.base_detection_enu_failure_count,
                    "association_updates": self.association_update_count,
                    "association_input_detections": self.association_input_detection_count,
                    "association_matches": self.association_match_count,
                    "association_unmatched_detections": self.association_unmatched_detection_count,
                    "association_unmatched_tracks": self.association_unmatched_track_count,
                    "tentative_track_observations": self.tracker_tentative_observation_count,
                    "confirmed_track_observations": self.tracker_confirmed_observation_count,
                    "tentative_track_peak": self.tracker_tentative_peak,
                    "confirmed_track_peak": self.tracker_confirmed_peak,
                    "track_births": self.tracker_spawn_count,
                    "track_deletions": self.tracker_deleted_count,
                    "final_published_objects": self.dynamic_track_count,
                    "drop_reasons": {
                        "base_detection": dict(sorted(self.base_detection_drop_reasons.items())),
                        "dynamic_filter": dict(sorted(self.dynamic_filter_drop_reasons.items())),
                    },
                },
                "velocity_frame": PUBLIC_FRAME,
                "velocity_semantics": "ABSOLUTE_SAME_FRAME",
                "publish_speed_gate_mps": self.dynamic_publish_speed_gate_mps,
                "velocity_publication_policy": "NO_SPEED_MAGNITUDE_GATE",
                "minimum_candidate_enu_z_m": self.minimum_candidate_enu_z_m,
                "minimum_public_hit_count": self.minimum_public_hit_count,
                "semantic_filter": {
                    "enabled": self.semantic_filter_enabled,
                    "model_path": self.semantic_model_path,
                    "model_metadata": None if self.semantic_model is None else self.semantic_model.metadata,
                    "primary_minimum_hit_count": self.semantic_primary_minimum_hit_count,
                    "scored_tracks": self.semantic_scored_track_count,
                    "high_probability_tracks": self.semantic_high_probability_track_count,
                    "motion_supported_tracks": self.semantic_motion_supported_track_count,
                    "history_count_peak": self.semantic_history_count_peak,
                    "object_memory": (
                        None
                        if self.semantic_object_memory is None
                        else self.semantic_object_memory.status()
                    ),
                    "background_map": (
                        None
                        if self.semantic_background_map is None
                        else self.semantic_background_map.status()
                    ),
                    "maximum_background_frames": self.semantic_maximum_background_frames,
                    "velocity_refinement": {
                        "enabled": self.semantic_velocity_refiner is not None,
                        "refined_outputs": (
                            0 if self.semantic_velocity_refiner is None
                            else self.semantic_velocity_refiner.refined_count
                        ),
                        "arc_coasting_outputs": (
                            0 if self.semantic_velocity_refiner is None
                            else self.semantic_velocity_refiner.arc_coast_count
                        ),
                    },
                    "fleet_speed_calibration": (
                        None
                        if self.semantic_fleet_speed_calibrator is None
                        else self.semantic_fleet_speed_calibrator.status()
                    ),
                    "production_truth_input": False,
                },
                "measurement_floor_m": self.tracker_measurement_floor_m,
                "partial_surface_std_m": self.tracker_partial_surface_std_m,
                "minimum_entity_evidence_hits": self.tracker_minimum_entity_evidence_hits,
                "require_entity_evidence_for_spawn": self.tracker_require_entity_evidence_for_spawn,
                "velocity_valid_definition": (
                    "finite vx/vy/speed, scalar consistency, hit_count >= "
                    f"{self.semantic_primary_minimum_hit_count if self.semantic_filter_enabled else self.minimum_public_hit_count}"
                ),
                "velocity_valid_output_count": self.velocity_valid_output_count,
                "velocity_measured_output_count": self.velocity_measured_output_count,
                "velocity_coasting_output_count": self.velocity_coasting_output_count,
                "velocity_invalid_output_count": self.velocity_invalid_output_count,
                "tracker": {
                    "constant_velocity_kalman": True,
                    "gated_one_to_one_association": True,
                    "birth_confirm_coast_delete": True,
                    "stable_track_id": True,
                    "gt_initialization": False,
                },
            },
            'dynamic_scene_evaluation': {
                "status": (
                    "DYNAMIC_TRACK_OUTPUT_EMPTY_CAUSE_UNRESOLVED"
                    if self.dynamic_track_count == 0
                    else "MEASURED_PENDING_INDEPENDENT_SCENE_EVALUATION"
                ),
                "production_gt_feedback": False,
                "note": "生产节点不读取 EVALUATION_ONLY truth；场景存在性、运动性和 LiDAR 可观测性由独立 evaluator 单独判定。",
            },
            'control_publishers': 0,
            'elapsed_wall_s': time.monotonic() - self.started_monotonic,
        }

    def write_receipt(self) -> None:
        if not self.receipt_path:
            return
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_path.write_text(
            json.dumps(self.build_receipt(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: S3LocalizationDynamicTracks | None = None
    try:
        node = S3LocalizationDynamicTracks()
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        # SIGTERM 触发 rclpy 外部关闭时，正常收口而不伪造运行失败。
        pass
    finally:
        if node is not None:
            node.write_receipt()
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
