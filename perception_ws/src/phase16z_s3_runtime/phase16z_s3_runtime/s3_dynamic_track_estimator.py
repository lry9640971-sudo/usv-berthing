"""S3 绝对公共坐标动态目标 tracker。"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection2D:
    x: float
    y: float
    radius_m: float = 1.5
    confidence: float = 1.0


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

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def heading_rad(self) -> float:
        return math.atan2(self.vy, self.vx) if self.speed_mps > 1e-9 else 0.0


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


def _assignment(costs: Sequence[Sequence[float | None]], unmatched_cost: float) -> tuple[tuple[int, int], ...]:
    """小目标数下的精确一对一关联，超过 12 个时退化为确定性贪心。"""

    track_count = len(costs)
    detection_count = len(costs[0]) if track_count else 0
    if track_count == 0 or detection_count == 0:
        return ()
    if track_count > 12 or detection_count > 12:
        candidates = sorted(
            (float(cost), track_index, detection_index)
            for track_index, row in enumerate(costs)
            for detection_index, cost in enumerate(row)
            if cost is not None and cost < unmatched_cost
        )
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        result = []
        for _, track_index, detection_index in candidates:
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            result.append((track_index, detection_index))
        return tuple(sorted(result))

    @functools.lru_cache(maxsize=None)
    def solve(track_index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if track_index == track_count:
            return 0.0, ()
        best_cost, best_pairs = solve(track_index + 1, used_mask)
        best_cost += unmatched_cost
        for detection_index, cost in enumerate(costs[track_index]):
            if cost is None or used_mask & (1 << detection_index):
                continue
            tail_cost, tail_pairs = solve(track_index + 1, used_mask | (1 << detection_index))
            candidate = float(cost) + tail_cost
            pairs = ((track_index, detection_index),) + tail_pairs
            if candidate < best_cost - 1e-12 or (math.isclose(candidate, best_cost, abs_tol=1e-12) and pairs < best_pairs):
                best_cost, best_pairs = candidate, pairs
        return best_cost, best_pairs

    return tuple(sorted(solve(0, 0)[1]))


class DynamicTrackEstimator:
    """常速度 Kalman、门控关联和 birth/confirm/coast/delete 生命周期。"""

    def __init__(
        self,
        frame_id: str,
        *,
        association_gate_m: float = 4.0,
        process_accel_std_mps2: float = 1.5,
        measurement_std_m: float = 0.35,
        confirmation_hits: int = 3,
        maximum_track_age_s: float = 1.0,
        maximum_missed_updates: int = 5,
    ) -> None:
        self.frame_id = str(frame_id).strip()
        if not self.frame_id:
            raise ValueError("TRACK_FRAME_MISSING")
        self.association_gate_m = float(association_gate_m)
        self.process_accel_std_mps2 = float(process_accel_std_mps2)
        self.measurement_std_m = float(measurement_std_m)
        self.confirmation_hits = int(confirmation_hits)
        self.maximum_track_age_s = float(maximum_track_age_s)
        self.maximum_missed_updates = int(maximum_missed_updates)
        self.tracks: dict[int, _Track] = {}
        self.next_track_id = 1
        self.last_stamp_ns: int | None = None
        self.last_association_match_count = 0
        self.last_unmatched_track_count = 0
        self.last_unmatched_detection_count = 0
        self.last_spawn_count = 0
        self.last_deleted_count = 0
        self.last_tentative_count = 0
        self.last_confirmed_count = 0

    def _predict(self, track: _Track, stamp_ns: int) -> None:
        raw_dt = (stamp_ns - track.last_state_stamp_ns) / 1_000_000_000.0
        if raw_dt <= 0.0:
            return
        dt = min(raw_dt, 1.0)
        transition = np.asarray(((1.0, 0.0, dt, 0.0), (0.0, 1.0, 0.0, dt), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)
        q = self.process_accel_std_mps2**2
        process = q * np.asarray(((dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0), (0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0), (dt**3 / 2.0, 0.0, dt**2, 0.0), (0.0, dt**3 / 2.0, 0.0, dt**2)), dtype=np.float64)
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process
        track.last_state_stamp_ns = stamp_ns

    def _correct(self, track: _Track, detection: Detection2D, stamp_ns: int) -> None:
        observation = np.asarray((detection.x, detection.y), dtype=np.float64)
        measurement = np.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)), dtype=np.float64)
        noise = np.eye(2, dtype=np.float64) * self.measurement_std_m**2
        innovation = observation - measurement @ track.state
        innovation_covariance = measurement @ track.covariance @ measurement.T + noise
        gain = track.covariance @ measurement.T @ np.linalg.inv(innovation_covariance)
        track.state = track.state + gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        residual = identity - gain @ measurement
        track.covariance = residual @ track.covariance @ residual.T + gain @ noise @ gain.T
        track.last_measurement_stamp_ns = stamp_ns
        track.hit_count += 1
        track.missed_count = 0
        track.radius_m = 0.75 * track.radius_m + 0.25 * float(detection.radius_m)
        track.confidence = 0.75 * track.confidence + 0.25 * float(detection.confidence)

    def _spawn(self, detection: Detection2D, stamp_ns: int) -> None:
        variance = self.measurement_std_m**2
        self.tracks[self.next_track_id] = _Track(
            track_id=self.next_track_id,
            state=np.asarray((detection.x, detection.y, 0.0, 0.0), dtype=np.float64),
            covariance=np.diag((variance, variance, 16.0, 16.0)).astype(np.float64),
            last_state_stamp_ns=stamp_ns,
            last_measurement_stamp_ns=stamp_ns,
            hit_count=1,
            missed_count=0,
            radius_m=float(detection.radius_m),
            confidence=float(detection.confidence),
        )
        self.next_track_id += 1

    def update(self, stamp_ns: int, detections: Iterable[Detection2D], *, frame_id: str) -> tuple[TrackEstimate, ...]:
        stamp = int(stamp_ns)
        if stamp <= 0 or str(frame_id).strip() != self.frame_id:
            raise ValueError("TRACK_INPUT_FRAME_OR_STAMP_INVALID")
        if self.last_stamp_ns is not None and stamp <= self.last_stamp_ns:
            raise ValueError("TRACK_STAMP_NOT_INCREASING")
        detection_list = tuple(detections)
        ordered = [self.tracks[key] for key in sorted(self.tracks)]
        for track in ordered:
            self._predict(track, stamp)
        costs: list[list[float | None]] = []
        for track in ordered:
            row = []
            for detection in detection_list:
                distance = math.hypot(track.state[0] - detection.x, track.state[1] - detection.y)
                row.append(distance if distance <= self.association_gate_m else None)
            costs.append(row)
        matches = _assignment(costs, unmatched_cost=self.association_gate_m * 1.05)
        self.last_association_match_count = len(matches)
        self.last_unmatched_track_count = len(ordered) - len(matches)
        matched_tracks = {index for index, _ in matches}
        matched_detections = {index for _, index in matches}
        for track_index, detection_index in matches:
            self._correct(ordered[track_index], detection_list[detection_index], stamp)
        for track_index, track in enumerate(ordered):
            if track_index not in matched_tracks:
                track.missed_count += 1
        self.last_unmatched_detection_count = len(detection_list) - len(matches)
        self.last_spawn_count = 0
        for detection_index, detection in enumerate(detection_list):
            if detection_index not in matched_detections:
                self._spawn(detection, stamp)
                self.last_spawn_count += 1
        expiry_ns = int(self.maximum_track_age_s * 1_000_000_000)
        track_count_before_expiry = len(self.tracks)
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if stamp - track.last_measurement_stamp_ns <= expiry_ns
            and track.missed_count <= self.maximum_missed_updates
        }
        self.last_deleted_count = track_count_before_expiry - len(self.tracks)
        self.last_tentative_count = sum(
            1 for track in self.tracks.values() if track.hit_count < self.confirmation_hits
        )
        self.last_confirmed_count = len(self.tracks) - self.last_tentative_count
        self.last_stamp_ns = stamp
        return self.estimates(stamp)

    def estimates(self, stamp_ns: int | None = None, *, include_tentative: bool = False) -> tuple[TrackEstimate, ...]:
        stamp = int(stamp_ns if stamp_ns is not None else (self.last_stamp_ns or 0))
        result = []
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            if not include_tentative and track.hit_count < self.confirmation_hits:
                continue
            result.append(
                TrackEstimate(
                    track_id=track.track_id,
                    x=float(track.state[0]),
                    y=float(track.state[1]),
                    vx=float(track.state[2]),
                    vy=float(track.state[3]),
                    radius_m=float(track.radius_m),
                    confidence=float(track.confidence),
                    age_sec=max(0.0, (stamp - track.last_measurement_stamp_ns) / 1_000_000_000.0),
                    covariance=tuple(float(value) for value in track.covariance.reshape(-1)),
                    hit_count=track.hit_count,
                    missed_count=track.missed_count,
                )
            )
        return tuple(result)


__all__ = ["Detection2D", "DynamicTrackEstimator", "TrackEstimate"]
