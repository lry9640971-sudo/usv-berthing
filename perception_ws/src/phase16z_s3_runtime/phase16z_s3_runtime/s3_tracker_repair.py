"""S3 绝对坐标 tracker 修复核心。

实现目标是让关联规模与目标数量无关：所有合法边进入一次 Hungarian
最小代价一对一关联，未匹配轨迹和未匹配观测由显式 dummy 列表示，绝不
在目标数大于 12 时偷偷退化为贪心。门控使用观测不确定度和预测协方差，
轨迹生命周期、coasting 和动态分类均单独记录。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


ASSIGNMENT_BACKEND = "scipy.optimize.linear_sum_assignment"


class TrackLifecycle(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    COASTING = "COASTING"
    DELETED = "DELETED"


@dataclass(frozen=True)
class TrackEstimate:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    radius_m: float
    confidence: float
    age_sec: float
    covariance: tuple[float, ...]
    hit_count: int
    missed_count: int
    lifecycle: str
    classification: str
    position_covariance: tuple[float, ...]
    velocity_covariance: tuple[float, ...]
    source_sensors: tuple[str, ...]

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def heading_rad(self) -> float:
        return math.atan2(self.vy, self.vx) if self.speed_mps > 1e-9 else 0.0


@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[tuple[int, int], ...]
    unmatched_track_indices: tuple[int, ...]
    unmatched_detection_indices: tuple[int, ...]
    cost_matrix_shape: tuple[int, int]
    backend: str = ASSIGNMENT_BACKEND


@dataclass
class _Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    last_state_stamp_ns: int
    last_measurement_stamp_ns: int
    hit_count: int
    missed_count: int
    radius_m: float
    confidence: float
    lifecycle: TrackLifecycle
    source_sensors: tuple[str, ...]
    dynamic_evidence_sec: float = 0.0
    dynamic_confirmed: bool = False
    entity_evidence_hits: int = 0


def _observation_value(observation: Any, field_name: str, default: float) -> float:
    try:
        return float(getattr(observation, field_name))
    except (AttributeError, TypeError, ValueError):
        return float(default)


def _observation_sources(observation: Any) -> tuple[str, ...]:
    sources = getattr(observation, "source_sensors", ())
    if isinstance(sources, str):
        return (sources,)
    result = tuple(sorted({str(item) for item in sources if str(item)}))
    return result or ("unknown",)


def exact_one_to_one_assignment(
    cost_rows: Sequence[Sequence[float | None]],
    *,
    detection_count: int | None = None,
    unmatched_cost: float = 12.0,
) -> AssociationResult:
    """带 dummy 未匹配列的精确全局一对一关联。"""

    track_count = len(cost_rows)
    if detection_count is None:
        detection_count = len(cost_rows[0]) if track_count else 0
    detection_count = int(detection_count)
    if detection_count < 0:
        raise ValueError("ASSOCIATION_DETECTION_COUNT_INVALID")
    if track_count and any(len(row) != detection_count for row in cost_rows):
        raise ValueError("ASSOCIATION_COST_MATRIX_RAGGED")
    if track_count == 0:
        return AssociationResult((), (), tuple(range(detection_count)), (0, detection_count))
    if detection_count == 0:
        return AssociationResult((), tuple(range(track_count)), (), (track_count, 0))
    if not math.isfinite(float(unmatched_cost)) or float(unmatched_cost) <= 0.0:
        raise ValueError("ASSOCIATION_UNMATCHED_COST_INVALID")

    invalid_cost = float(unmatched_cost) + 1_000_000.0
    matrix = np.full((track_count, detection_count + track_count), invalid_cost, dtype=np.float64)
    for track_index, row in enumerate(cost_rows):
        for detection_index, value in enumerate(row):
            if value is not None and math.isfinite(float(value)):
                matrix[track_index, detection_index] = float(value) + 1e-9 * (track_index + detection_index)
        for dummy_index in range(track_count):
            matrix[track_index, detection_count + dummy_index] = float(unmatched_cost) + 1e-9 * abs(track_index - dummy_index)

    row_indices, column_indices = linear_sum_assignment(matrix)
    pairs: list[tuple[int, int]] = []
    unmatched_tracks: list[int] = []
    matched_detections: set[int] = set()
    for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
        if column_index < detection_count and matrix[row_index, column_index] < invalid_cost:
            pairs.append((int(row_index), int(column_index)))
            matched_detections.add(int(column_index))
        else:
            unmatched_tracks.append(int(row_index))
    pairs.sort()
    unmatched_tracks.sort()
    unmatched_detections = tuple(index for index in range(detection_count) if index not in matched_detections)
    return AssociationResult(
        matches=tuple(pairs),
        unmatched_track_indices=tuple(unmatched_tracks),
        unmatched_detection_indices=unmatched_detections,
        cost_matrix_shape=(track_count, detection_count + track_count),
    )


class RepairedTrackEstimator:
    """常速度 Kalman + 协方差门控 + 显式生命周期的 tracker。"""

    def __init__(
        self,
        frame_id: str,
        *,
        association_gate_chi2: float = 9.21,
        process_accel_std_mps2: float = 1.5,
        initial_position_std_m: float = 0.7,
        initial_velocity_std_mps: float = 4.0,
        confirmation_hits: int = 3,
        maximum_missed_updates: int = 5,
        maximum_track_age_s: float = 1.0,
        dynamic_speed_threshold_mps: float = 0.15,
        dynamic_persistence_s: float = 0.30,
        measurement_floor_m: float = 0.05,
        partial_surface_std_m: float = 0.0,
        motion_model: str = "constant_velocity",
        process_jerk_std_mps3: float = 0.6,
        initial_acceleration_std_mps2: float = 1.0,
        minimum_entity_evidence_hits: int = 0,
        require_entity_evidence_for_spawn: bool = False,
        latch_dynamic_classification: bool = False,
    ) -> None:
        self.frame_id = str(frame_id).strip()
        if not self.frame_id:
            raise ValueError("TRACK_FRAME_MISSING")
        self.association_gate_chi2 = float(association_gate_chi2)
        self.process_accel_std_mps2 = float(process_accel_std_mps2)
        self.initial_position_std_m = float(initial_position_std_m)
        self.initial_velocity_std_mps = float(initial_velocity_std_mps)
        self.confirmation_hits = int(confirmation_hits)
        self.maximum_missed_updates = int(maximum_missed_updates)
        self.maximum_track_age_s = float(maximum_track_age_s)
        self.dynamic_speed_threshold_mps = float(dynamic_speed_threshold_mps)
        self.dynamic_persistence_s = float(dynamic_persistence_s)
        self.measurement_floor_m = float(measurement_floor_m)
        self.partial_surface_std_m = float(partial_surface_std_m)
        self.motion_model = str(motion_model).strip()
        self.process_jerk_std_mps3 = float(process_jerk_std_mps3)
        self.initial_acceleration_std_mps2 = float(initial_acceleration_std_mps2)
        self.minimum_entity_evidence_hits = int(minimum_entity_evidence_hits)
        self.require_entity_evidence_for_spawn = bool(require_entity_evidence_for_spawn)
        self.latch_dynamic_classification = bool(latch_dynamic_classification)
        if self.confirmation_hits < 1 or self.maximum_missed_updates < 0 or self.maximum_track_age_s <= 0.0:
            raise ValueError("TRACK_LIFECYCLE_PARAMETER_INVALID")
        if self.motion_model not in ("constant_velocity", "constant_acceleration"):
            raise ValueError("TRACK_MOTION_MODEL_INVALID")
        if self.process_jerk_std_mps3 <= 0.0 or self.initial_acceleration_std_mps2 <= 0.0:
            raise ValueError("TRACK_ACCELERATION_PARAMETER_INVALID")
        if self.minimum_entity_evidence_hits < 0:
            raise ValueError("TRACK_ENTITY_EVIDENCE_PARAMETER_INVALID")
        self.tracks: dict[int, _Track] = {}
        self.next_track_id = 1
        self.last_stamp_ns: int | None = None
        self.last_association = AssociationResult((), (), (), (0, 0))
        self.last_deleted_count = 0
        self.last_spawn_count = 0
        self.last_rejected_spawn_count = 0

    def _transition(self, dt: float) -> np.ndarray:
        if self.motion_model == "constant_acceleration":
            return np.asarray(
                (
                    (1.0, 0.0, dt, 0.0, dt**2 / 2.0, 0.0),
                    (0.0, 1.0, 0.0, dt, 0.0, dt**2 / 2.0),
                    (0.0, 0.0, 1.0, 0.0, dt, 0.0),
                    (0.0, 0.0, 0.0, 1.0, 0.0, dt),
                    (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            )
        return np.asarray(
            ((1.0, 0.0, dt, 0.0), (0.0, 1.0, 0.0, dt), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    def _predict(self, track: _Track, stamp_ns: int) -> None:
        raw_dt = (int(stamp_ns) - int(track.last_state_stamp_ns)) / 1_000_000_000.0
        if raw_dt <= 0.0:
            return
        dt = min(raw_dt, 5.0)
        transition = self._transition(dt)
        if self.motion_model == "constant_acceleration":
            process = np.zeros((6, 6), dtype=np.float64)
            one_axis = float(self.process_jerk_std_mps3) ** 2 * np.asarray(
                (
                    (dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0),
                    (dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0),
                    (dt**3 / 6.0, dt**2 / 2.0, dt),
                ),
                dtype=np.float64,
            )
            for axis in range(2):
                indices = np.asarray((axis, axis + 2, axis + 4))
                process[np.ix_(indices, indices)] = one_axis
        else:
            q = float(self.process_accel_std_mps2) ** 2
            process = q * np.asarray(
                ((dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0), (0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0), (dt**3 / 2.0, 0.0, dt**2, 0.0), (0.0, dt**3 / 2.0, 0.0, dt**2)),
                dtype=np.float64,
            )
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process
        track.covariance = 0.5 * (track.covariance + track.covariance.T)
        track.last_state_stamp_ns = int(stamp_ns)

    def _measurement_noise(self, observation: Any) -> np.ndarray:
        std = max(0.05, min(5.0, _observation_value(observation, "position_std_m", 0.7)))
        # 稀疏可见表面不是目标中心的精确测量；误差下界随回波数降低。
        support = max(1.0, _observation_value(observation, "support", 1.0))
        std = max(std, self.measurement_floor_m, self.partial_surface_std_m / math.sqrt(support))
        return np.eye(2, dtype=np.float64) * std**2

    def _innovation(self, track: _Track, observation: Any) -> tuple[np.ndarray, np.ndarray, float]:
        measurement = np.zeros((2, len(track.state)), dtype=np.float64)
        measurement[0, 0] = measurement[1, 1] = 1.0
        vector = np.asarray((_observation_value(observation, "x_m", math.nan), _observation_value(observation, "y_m", math.nan)), dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError("TRACK_DETECTION_NONFINITE")
        residual = vector - measurement @ track.state
        covariance = measurement @ track.covariance @ measurement.T + self._measurement_noise(observation)
        covariance = 0.5 * (covariance + covariance.T)
        try:
            solved = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError as exc:
            raise ValueError("TRACK_INNOVATION_COVARIANCE_SINGULAR") from exc
        mahalanobis = float(residual.T @ solved)
        return residual, covariance, mahalanobis

    def _cost(self, track: _Track, observation: Any) -> float | None:
        _, _, mahalanobis = self._innovation(track, observation)
        if mahalanobis > self.association_gate_chi2:
            return None
        radius_track = max(0.05, float(track.radius_m))
        radius_observation = max(0.05, _observation_value(observation, "radius_m", radius_track))
        size_penalty = min(4.0, abs(math.log(radius_observation / radius_track)))
        confidence_penalty = 0.5 * (1.0 - min(0.99, max(0.01, _observation_value(observation, "confidence", 0.2))))
        return mahalanobis + 0.25 * size_penalty + confidence_penalty

    def _association_costs(self, tracks, observations):
        """批量计算同一协方差门，避免每条边重复分配小矩阵。"""
        if not observations:
            return [[] for _ in tracks]
        xy = np.asarray([[_observation_value(o, 'x_m', math.nan), _observation_value(o, 'y_m', math.nan)] for o in observations])
        if not np.all(np.isfinite(xy)):
            raise ValueError('TRACK_DETECTION_NONFINITE')
        variance = np.asarray([self._measurement_noise(o)[0, 0] for o in observations])
        radii = np.asarray([max(.05, _observation_value(o, 'radius_m', .5)) for o in observations])
        penalty = np.asarray([.5 * (1. - min(.99, max(.01, _observation_value(o, 'confidence', .2)))) for o in observations])
        rows = []
        for track in tracks:
            dx, dy = (xy - track.state[:2]).T
            a = track.covariance[0, 0] + variance
            b = (track.covariance[0, 1] + track.covariance[1, 0]) / 2
            d = track.covariance[1, 1] + variance
            mahalanobis = (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / (a * d - b * b)
            size = np.minimum(4., np.abs(np.log(radii / max(.05, track.radius_m))))
            costs = mahalanobis + .25 * size + penalty
            rows.append([float(cost) if distance <= self.association_gate_chi2 else None
                         for cost, distance in zip(costs, mahalanobis)])
        return rows

    def _correct(self, track: _Track, observation: Any, stamp_ns: int) -> None:
        measurement = np.zeros((2, len(track.state)), dtype=np.float64)
        measurement[0, 0] = measurement[1, 1] = 1.0
        residual, innovation_covariance, _ = self._innovation(track, observation)
        gain = track.covariance @ measurement.T @ np.linalg.inv(innovation_covariance)
        track.state = track.state + gain @ residual
        identity = np.eye(len(track.state), dtype=np.float64)
        residual_operator = identity - gain @ measurement
        noise = self._measurement_noise(observation)
        track.covariance = residual_operator @ track.covariance @ residual_operator.T + gain @ noise @ gain.T
        track.covariance = 0.5 * (track.covariance + track.covariance.T)
        track.last_measurement_stamp_ns = int(stamp_ns)
        track.hit_count += 1
        track.missed_count = 0
        track.lifecycle = TrackLifecycle.CONFIRMED if track.hit_count >= self.confirmation_hits else TrackLifecycle.TENTATIVE
        radius_observation = max(0.05, _observation_value(observation, "radius_m", track.radius_m))
        track.radius_m = 0.75 * float(track.radius_m) + 0.25 * radius_observation
        observation_confidence = min(0.99, max(0.01, _observation_value(observation, "confidence", 0.2)))
        support = max(1.0, _observation_value(observation, "support", 1.0))
        support_term = min(0.95, support / (support + 3.0))
        uncertainty_term = math.exp(-max(0.0, _observation_value(observation, "position_std_m", 0.7)) / 2.0)
        evidence_confidence = max(0.02, min(0.98, observation_confidence * support_term * uncertainty_term))
        track.confidence = max(0.02, min(0.98, 0.65 * track.confidence + 0.35 * evidence_confidence))
        track.source_sensors = tuple(sorted(set(track.source_sensors) | set(_observation_sources(observation))))
        if bool(getattr(observation, "entity_evidence", False)):
            track.entity_evidence_hits += 1

    def _spawn(self, observation: Any, stamp_ns: int) -> None:
        x = _observation_value(observation, "x_m", math.nan)
        y = _observation_value(observation, "y_m", math.nan)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("TRACK_DETECTION_NONFINITE")
        position_std = math.sqrt(float(self._measurement_noise(observation)[0, 0]))
        confidence = min(0.80, max(0.05, _observation_value(observation, "confidence", 0.2)))
        if self.motion_model == "constant_acceleration":
            state = np.asarray((x, y, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)
            covariance = np.diag((position_std**2, position_std**2,
                                  self.initial_velocity_std_mps**2, self.initial_velocity_std_mps**2,
                                  self.initial_acceleration_std_mps2**2,
                                  self.initial_acceleration_std_mps2**2)).astype(np.float64)
        else:
            state = np.asarray((x, y, 0.0, 0.0), dtype=np.float64)
            covariance = np.diag((position_std**2, position_std**2,
                                  self.initial_velocity_std_mps**2,
                                  self.initial_velocity_std_mps**2)).astype(np.float64)
        track = _Track(
            track_id=self.next_track_id,
            state=state,
            covariance=covariance,
            last_state_stamp_ns=int(stamp_ns),
            last_measurement_stamp_ns=int(stamp_ns),
            hit_count=1,
            missed_count=0,
            radius_m=max(0.05, _observation_value(observation, "radius_m", 1.0)),
            confidence=confidence,
            lifecycle=TrackLifecycle.CONFIRMED if self.confirmation_hits <= 1 else TrackLifecycle.TENTATIVE,
            source_sensors=_observation_sources(observation),
            entity_evidence_hits=int(bool(getattr(observation, "entity_evidence", False))),
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def _update_dynamic_evidence(
        self,
        track: _Track,
        stamp_ns: int,
        *,
        matched: bool,
        previous_measurement_stamp_ns: int,
    ) -> None:
        if not matched or track.lifecycle != TrackLifecycle.CONFIRMED:
            track.dynamic_evidence_sec = max(0.0, track.dynamic_evidence_sec - 0.05)
            return
        speed = math.hypot(float(track.state[2]), float(track.state[3]))
        velocity_covariance = track.covariance[2:4, 2:4]
        speed_sigma = math.sqrt(max(1e-9, float(np.trace(velocity_covariance) / 2.0)))
        lower_speed_bound = speed - 2.0 * speed_sigma
        elapsed = max(0.0, (int(stamp_ns) - int(previous_measurement_stamp_ns)) / 1_000_000_000.0)
        if lower_speed_bound >= self.dynamic_speed_threshold_mps:
            track.dynamic_evidence_sec += elapsed
        else:
            track.dynamic_evidence_sec = max(0.0, track.dynamic_evidence_sec - max(0.05, elapsed))
        if track.dynamic_evidence_sec >= self.dynamic_persistence_s:
            track.dynamic_confirmed = True

    def update(self, stamp_ns: int, observations: Iterable[Any], *, frame_id: str) -> tuple[TrackEstimate, ...]:
        stamp = int(stamp_ns)
        if stamp <= 0 or str(frame_id).strip() != self.frame_id:
            raise ValueError("TRACK_INPUT_FRAME_OR_STAMP_INVALID")
        if self.last_stamp_ns is not None and stamp <= int(self.last_stamp_ns):
            raise ValueError("TRACK_STAMP_NOT_INCREASING")
        observation_list = tuple(observations)
        for observation in observation_list:
            observation_stamp = getattr(observation, "stamp_ns", stamp)
            if int(observation_stamp) != stamp:
                raise ValueError("TRACK_OBSERVATION_STAMP_MISMATCH")

        # 必须先过期再关联，不能让新观测把已失效轨迹的测量时间刷新。
        expiry_ns = int(self.maximum_track_age_s * 1_000_000_000)
        expired = [key for key, track in self.tracks.items()
                   if stamp - track.last_measurement_stamp_ns > expiry_ns]
        for key in expired:
            self.tracks[key].lifecycle = TrackLifecycle.DELETED
            del self.tracks[key]
        ordered_tracks = [self.tracks[key] for key in sorted(self.tracks)]
        previous_measurement_stamps = {
            track.track_id: int(track.last_measurement_stamp_ns)
            for track in ordered_tracks
        }
        for track in ordered_tracks:
            self._predict(track, stamp)
        costs = self._association_costs(ordered_tracks, observation_list)
        self.last_association = exact_one_to_one_assignment(costs, detection_count=len(observation_list))
        matched_tracks = {track_index for track_index, _ in self.last_association.matches}
        matched_track_ids = {ordered_tracks[index].track_id for index in matched_tracks}
        matched_observations = {observation_index for _, observation_index in self.last_association.matches}
        for track_index, observation_index in self.last_association.matches:
            self._correct(ordered_tracks[track_index], observation_list[observation_index], stamp)
        for track_index, track in enumerate(ordered_tracks):
            if track_index not in matched_tracks:
                track.missed_count += 1
                track.lifecycle = (
                    TrackLifecycle.COASTING
                    if track.lifecycle in (TrackLifecycle.CONFIRMED, TrackLifecycle.COASTING)
                    else TrackLifecycle.TENTATIVE
                )
                track.confidence = max(0.02, 0.90 * track.confidence)
        self.last_spawn_count = 0
        self.last_rejected_spawn_count = 0
        for observation_index, observation in enumerate(observation_list):
            if observation_index not in matched_observations:
                if (
                    self.require_entity_evidence_for_spawn
                    and not bool(getattr(observation, "entity_evidence", False))
                ):
                    self.last_rejected_spawn_count += 1
                    continue
                self._spawn(observation, stamp)
                self.last_spawn_count += 1

        for track in self.tracks.values():
            self._update_dynamic_evidence(
                track,
                stamp,
                matched=track.track_id in matched_track_ids,
                previous_measurement_stamp_ns=previous_measurement_stamps.get(track.track_id, stamp),
            )

        expiry_ns = int(self.maximum_track_age_s * 1_000_000_000)
        before_count = len(self.tracks)
        survivors: dict[int, _Track] = {}
        for track_id, track in self.tracks.items():
            age_ns = stamp - int(track.last_measurement_stamp_ns)
            if age_ns <= expiry_ns and track.missed_count <= self.maximum_missed_updates:
                survivors[track_id] = track
            else:
                track.lifecycle = TrackLifecycle.DELETED
        self.tracks = survivors
        self.last_deleted_count = len(expired) + before_count - len(self.tracks)
        self.last_stamp_ns = stamp
        return self.estimates(stamp, include_coasting=True)

    def _classification(self, track: _Track) -> str:
        # dynamic_tracks 只表示动态目标；静态岸线、码头和背景聚类必须留在
        # 内部关联状态中，不能越过公共输出边界。短暂丢帧时保留已有动态证据，
        # 允许动态目标进入 COASTING 状态继续输出。
        if (
            track.lifecycle in (TrackLifecycle.CONFIRMED, TrackLifecycle.COASTING)
            and (
                track.dynamic_evidence_sec >= self.dynamic_persistence_s
                or (self.latch_dynamic_classification and track.dynamic_confirmed)
            )
            and track.entity_evidence_hits >= self.minimum_entity_evidence_hits
        ):
            return "BOAT"
        return "UNKNOWN"

    def estimates(self, stamp_ns: int | None = None, *, include_tentative: bool = False, include_coasting: bool = True) -> tuple[TrackEstimate, ...]:
        stamp = int(stamp_ns if stamp_ns is not None else (self.last_stamp_ns or 0))
        result: list[TrackEstimate] = []
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            if track.lifecycle == TrackLifecycle.TENTATIVE and not include_tentative:
                continue
            if track.lifecycle == TrackLifecycle.COASTING and not include_coasting:
                continue
            position_covariance = tuple(float(value) for value in track.covariance[:2, :2].reshape(-1))
            velocity_covariance = tuple(float(value) for value in track.covariance[2:4, 2:4].reshape(-1))
            public_indices = np.asarray((0, 1, 2, 3))
            public_covariance = track.covariance[np.ix_(public_indices, public_indices)]
            result.append(
                TrackEstimate(
                    track_id=int(track.track_id),
                    x=float(track.state[0]),
                    y=float(track.state[1]),
                    vx=float(track.state[2]),
                    vy=float(track.state[3]),
                    radius_m=float(track.radius_m),
                    confidence=float(track.confidence),
                    age_sec=max(0.0, (stamp - int(track.last_measurement_stamp_ns)) / 1_000_000_000.0),
                    covariance=tuple(float(value) for value in public_covariance.reshape(-1)),
                    hit_count=int(track.hit_count),
                    missed_count=int(track.missed_count),
                    lifecycle=track.lifecycle.value,
                    classification=self._classification(track),
                    position_covariance=position_covariance,
                    velocity_covariance=velocity_covariance,
                    source_sensors=track.source_sensors,
                )
            )
        return tuple(result)

    def funnel(self) -> dict[str, Any]:
        tentative = sum(track.lifecycle == TrackLifecycle.TENTATIVE for track in self.tracks.values())
        confirmed = sum(track.lifecycle == TrackLifecycle.CONFIRMED for track in self.tracks.values())
        coasting = sum(track.lifecycle == TrackLifecycle.COASTING for track in self.tracks.values())
        return {
            "association_backend": ASSIGNMENT_BACKEND,
            "association_cost_matrix_shape": list(self.last_association.cost_matrix_shape),
            "association_matches": len(self.last_association.matches),
            "unmatched_tracks": len(self.last_association.unmatched_track_indices),
            "unmatched_detections": len(self.last_association.unmatched_detection_indices),
            "tentative_tracks": int(tentative),
            "confirmed_tracks": int(confirmed),
            "coasting_tracks": int(coasting),
            "track_births": int(self.last_spawn_count),
            "rejected_nonentity_births": int(self.last_rejected_spawn_count),
            "track_deletions": int(self.last_deleted_count),
            "final_published_objects": int(confirmed + coasting),
            "gt_initialization": False,
        }


__all__ = [
    "ASSIGNMENT_BACKEND",
    "AssociationResult",
    "RepairedTrackEstimator",
    "TrackEstimate",
    "TrackLifecycle",
    "exact_one_to_one_assignment",
]
