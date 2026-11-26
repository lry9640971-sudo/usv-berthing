"""S3 合法 GPS+IMU 定位的无 ROS 数学内核。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


WGS84_A_M = 6378137.0
WGS84_E2 = 6.6943799901413165e-3


def wrap_angle(angle_rad: float) -> float:
    """把角度归一化到 [-pi, pi)。"""

    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def unwrap_angle(previous_rad: float | None, current_rad: float) -> float:
    """对 IMU yaw 做最短弧展开。"""

    current = wrap_angle(current_rad)
    if previous_rad is None:
        return current
    return float(previous_rad) + wrap_angle(current - float(previous_rad))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """从合法单位四元数提取 ENU yaw。"""

    values = tuple(float(value) for value in (x, y, z, w))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("IMU_QUATERNION_NONFINITE")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-9:
        raise ValueError("IMU_QUATERNION_ZERO")
    x, y, z, w = (value / norm for value in values)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class WGS84Datum:
    """第一帧合格 NavSatFix 的 ECEF datum。"""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    ecef_x_m: float
    ecef_y_m: float
    ecef_z_m: float


class WGS84LocalENU:
    """WGS84 局部切平面，x=East、y=North、z=Up。"""

    def __init__(self, datum: WGS84Datum) -> None:
        self.datum = datum
        self._lat0 = math.radians(datum.latitude_deg)
        self._lon0 = math.radians(datum.longitude_deg)

    @staticmethod
    def _ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> tuple[float, float, float]:
        latitude = math.radians(float(latitude_deg))
        longitude = math.radians(float(longitude_deg))
        sine_latitude = math.sin(latitude)
        radius = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sine_latitude * sine_latitude)
        return (
            (radius + altitude_m) * math.cos(latitude) * math.cos(longitude),
            (radius + altitude_m) * math.cos(latitude) * math.sin(longitude),
            (radius * (1.0 - WGS84_E2) + altitude_m) * sine_latitude,
        )

    @classmethod
    def from_fix(cls, latitude_deg: float, longitude_deg: float, altitude_m: float) -> "WGS84LocalENU":
        values = tuple(float(value) for value in (latitude_deg, longitude_deg, altitude_m))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("GPS_DATUM_NONFINITE")
        if not -90.0 <= values[0] <= 90.0 or not -180.0 <= values[1] <= 180.0:
            raise ValueError("GPS_DATUM_OUT_OF_RANGE")
        ecef = cls._ecef(*values)
        return cls(WGS84Datum(*values, *ecef))

    def project(self, latitude_deg: float, longitude_deg: float, altitude_m: float) -> tuple[float, float, float]:
        values = tuple(float(value) for value in (latitude_deg, longitude_deg, altitude_m))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("GPS_FIX_NONFINITE")
        x, y, z = self._ecef(*values)
        dx, dy, dz = x - self.datum.ecef_x_m, y - self.datum.ecef_y_m, z - self.datum.ecef_z_m
        sine_latitude, cosine_latitude = math.sin(self._lat0), math.cos(self._lat0)
        sine_longitude, cosine_longitude = math.sin(self._lon0), math.cos(self._lon0)
        east = -sine_longitude * dx + cosine_longitude * dy
        north = (
            -sine_latitude * cosine_longitude * dx
            - sine_latitude * sine_longitude * dy
            + cosine_latitude * dz
        )
        up = (
            cosine_latitude * cosine_longitude * dx
            + cosine_latitude * sine_longitude * dy
            + sine_latitude * dz
        )
        return (east, north, up)


@dataclass(frozen=True)
class KalmanUpdate:
    accepted: bool
    mahalanobis_squared: float | None
    innovation_x_m: float
    innovation_y_m: float


class ConstantVelocityKalman2D:
    """二维常速度 Kalman filter，状态为 x、y、vx、vy。"""

    def __init__(self, *, process_accel_std_mps2: float = 1.5, gate_sigma: float = 4.0) -> None:
        if process_accel_std_mps2 <= 0.0 or gate_sigma <= 0.0:
            raise ValueError("KALMAN_CONFIG_INVALID")
        self.process_accel_std_mps2 = float(process_accel_std_mps2)
        self.gate_sigma = float(gate_sigma)
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.diag((4.0, 4.0, 25.0, 25.0)).astype(np.float64)
        self.last_stamp_ns: int | None = None
        self.initialized = False
        self.rejected_updates = 0

    def _predict_to(self, stamp_ns: int) -> None:
        if self.last_stamp_ns is None:
            return
        raw_dt = (int(stamp_ns) - self.last_stamp_ns) / 1_000_000_000.0
        if raw_dt <= 0.0:
            return
        dt = min(raw_dt, 2.0)
        transition = np.asarray(
            ((1.0, 0.0, dt, 0.0), (0.0, 1.0, 0.0, dt), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        q = self.process_accel_std_mps2**2
        process = q * np.asarray(
            ((dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0), (0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0), (dt**3 / 2.0, 0.0, dt**2, 0.0), (0.0, dt**3 / 2.0, 0.0, dt**2)),
            dtype=np.float64,
        )
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.last_stamp_ns = int(stamp_ns)

    def update(
        self,
        stamp_ns: int,
        position_xy_m: Iterable[float],
        covariance_xy_m2: Iterable[float] = (4.0, 4.0),
    ) -> KalmanUpdate:
        stamp = int(stamp_ns)
        if stamp <= 0:
            raise ValueError("KALMAN_STAMP_INVALID")
        position = tuple(float(value) for value in position_xy_m)
        if len(position) != 2 or not all(math.isfinite(value) for value in position):
            raise ValueError("KALMAN_POSITION_INVALID")
        measurement_covariance = tuple(float(value) for value in covariance_xy_m2)
        if len(measurement_covariance) != 2 or not all(math.isfinite(value) and value > 0.0 for value in measurement_covariance):
            measurement_covariance = (4.0, 4.0)
        if not self.initialized:
            self.state[:2] = position
            self.state[2:] = 0.0
            self.covariance = np.diag((*measurement_covariance, *([25.0] * (self.state.size - 2)))).astype(np.float64)
            self.last_stamp_ns = stamp
            self.initialized = True
            return KalmanUpdate(True, 0.0, 0.0, 0.0)

        self._predict_to(stamp)
        observation = np.asarray(position, dtype=np.float64)
        measurement = np.zeros((2, self.state.size), dtype=np.float64)
        measurement[:, :2] = np.eye(2)
        noise = np.diag(measurement_covariance).astype(np.float64)
        innovation = observation - measurement @ self.state
        innovation_covariance = measurement @ self.covariance @ measurement.T + noise
        try:
            inverse = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("KALMAN_INNOVATION_SINGULAR") from exc
        mahalanobis_squared = float(innovation.T @ inverse @ innovation)
        if mahalanobis_squared > self.gate_sigma**2:
            self.rejected_updates += 1
            return KalmanUpdate(False, mahalanobis_squared, float(innovation[0]), float(innovation[1]))
        gain = self.covariance @ measurement.T @ inverse
        self.state = self.state + gain @ innovation
        identity = np.eye(self.state.size, dtype=np.float64)
        residual = identity - gain @ measurement
        self.covariance = residual @ self.covariance @ residual.T + gain @ noise @ gain.T
        self.last_stamp_ns = stamp
        return KalmanUpdate(True, mahalanobis_squared, float(innovation[0]), float(innovation[1]))

    def update_velocity(self, velocity_xy_mps, variance_m2ps2):
        """融合同一 GPS 观测的实际速度，避免靠位置差分滞后推断制动。"""
        observation = np.asarray(tuple(velocity_xy_mps), dtype=float)
        if observation.shape != (2,) or not np.isfinite(observation).all() or variance_m2ps2 <= 0:
            raise ValueError("GPS_VELOCITY_INVALID")
        if not self.initialized:
            raise ValueError("GPS_POSITION_NOT_INITIALIZED")
        innovation_covariance = self.covariance[2:4, 2:4] + np.eye(2) * variance_m2ps2
        innovation = observation-self.state[2:4]
        if float(innovation@np.linalg.solve(innovation_covariance, innovation)) > self.gate_sigma**2:
            return False
        gain = self.covariance[:, 2:4] @ np.linalg.inv(innovation_covariance)
        self.state += gain @ innovation
        residual = np.eye(self.state.size)
        residual[:, 2:4] -= gain
        self.covariance = residual @ self.covariance @ residual.T + gain @ (np.eye(2) * variance_m2ps2) @ gain.T
        return True


class ConstantAccelerationKalman2D(ConstantVelocityKalman2D):
    """自船状态显式保留加速度，避免急制动反向时常速度估计仍停留在零附近。"""

    def __init__(self, *, process_jerk_power_m2ps5=25., gate_sigma=4.):
        super().__init__(gate_sigma=gate_sigma)
        if not math.isfinite(process_jerk_power_m2ps5) or process_jerk_power_m2ps5 <= 0:
            raise ValueError("KALMAN_JERK_POWER_INVALID")
        self.process_jerk_power_m2ps5 = float(process_jerk_power_m2ps5)
        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.diag((4., 4., 25., 25., 25., 25.))

    def _predict_to(self, stamp_ns):
        if self.last_stamp_ns is None:
            return
        raw_dt = (int(stamp_ns)-self.last_stamp_ns)/1e9
        if raw_dt <= 0:
            return
        dt = min(raw_dt, 2.)
        transition = np.eye(6)
        for axis in range(2):
            transition[axis, axis+2] = dt
            transition[axis, axis+4] = dt*dt/2
            transition[axis+2, axis+4] = dt
        block = self.process_jerk_power_m2ps5*np.asarray(
            ((dt**5/20, dt**4/8, dt**3/6),
             (dt**4/8, dt**3/3, dt**2/2),
             (dt**3/6, dt**2/2, dt)))
        process = np.zeros((6, 6))
        for axis in range(2):
            indices = [axis, axis+2, axis+4]
            process[np.ix_(indices, indices)] = block
        self.state = transition@self.state
        self.covariance = transition@self.covariance@transition.T+process
        self.last_stamp_ns = int(stamp_ns)


__all__ = [
    "ConstantAccelerationKalman2D",
    "ConstantVelocityKalman2D",
    "KalmanUpdate",
    "WGS84Datum",
    "WGS84LocalENU",
    "quaternion_to_yaw",
    "unwrap_angle",
    "wrap_angle",
]
