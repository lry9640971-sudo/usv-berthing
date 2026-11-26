"""S3 时间戳坐标变换工具。

这里把两类 TF 明确分开：传感器到 base 的刚性外参只能使用已经证明为
固定的变换；运动中的 ENU 与 base 关系必须按观测时间插值。模块不访问
Gazebo、真值或控制话题，可由 ROS 适配层调用。
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import math
from typing import Iterable, Sequence
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class RigidTransform2D:
    """固定 sensor->base 平面变换。"""

    translation_x_m: float
    translation_y_m: float
    yaw_rad: float
    source_frame: str
    target_frame: str
    is_model_proven_fixed: bool = False

    def __post_init__(self) -> None:
        if not self.source_frame or not self.target_frame:
            raise ValueError("SENSOR_TO_BASE_FRAME_MISSING")
        if not self.is_model_proven_fixed:
            raise ValueError("SENSOR_TO_BASE_FIXED_EXTERNAL_NOT_PROVEN")
        values = (self.translation_x_m, self.translation_y_m, self.yaw_rad)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("SENSOR_TO_BASE_TRANSFORM_NONFINITE")


@dataclass(frozen=True)
class RigidTransform3D:
    """固定 sensor->base 三维外参，LaserScan 平面点会保留 z=0 后完整旋转。"""

    translation_x_m: float
    translation_y_m: float
    translation_z_m: float
    quaternion_x: float
    quaternion_y: float
    quaternion_z: float
    quaternion_w: float
    source_frame: str
    target_frame: str
    is_model_proven_fixed: bool = False

    def __post_init__(self) -> None:
        if not self.source_frame or not self.target_frame:
            raise ValueError("SENSOR_TO_BASE_FRAME_MISSING")
        if not self.is_model_proven_fixed:
            raise ValueError("SENSOR_TO_BASE_FIXED_EXTERNAL_NOT_PROVEN")
        values = (
            self.translation_x_m,
            self.translation_y_m,
            self.translation_z_m,
            self.quaternion_x,
            self.quaternion_y,
            self.quaternion_z,
            self.quaternion_w,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("SENSOR_TO_BASE_TRANSFORM_NONFINITE")
        norm = math.sqrt(sum(float(value) ** 2 for value in (self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w)))
        if norm <= 1e-9:
            raise ValueError("SENSOR_TO_BASE_QUATERNION_INVALID")

    def rotate_translate(self, point_xyz_m: tuple[float, float, float]) -> tuple[float, float, float]:
        """用归一化四元数把 sensor 点映射到 base。"""

        norm = math.sqrt(sum(float(value) ** 2 for value in (self.quaternion_x, self.quaternion_y, self.quaternion_z, self.quaternion_w)))
        x = float(self.quaternion_x) / norm
        y = float(self.quaternion_y) / norm
        z = float(self.quaternion_z) / norm
        w = float(self.quaternion_w) / norm
        px, py, pz = (float(value) for value in point_xyz_m)
        # 四元数旋转的展开式，避免在 ROS 外部引入 geometry 依赖。
        rotated_x = (1.0 - 2.0 * (y * y + z * z)) * px + 2.0 * (x * y - z * w) * py + 2.0 * (x * z + y * w) * pz
        rotated_y = 2.0 * (x * y + z * w) * px + (1.0 - 2.0 * (x * x + z * z)) * py + 2.0 * (y * z - x * w) * pz
        rotated_z = 2.0 * (x * z - y * w) * px + 2.0 * (y * z + x * w) * py + (1.0 - 2.0 * (x * x + y * y)) * pz
        return (
            float(self.translation_x_m) + rotated_x,
            float(self.translation_y_m) + rotated_y,
            float(self.translation_z_m) + rotated_z,
        )


@dataclass(frozen=True)
class PlanarPose:
    """观测时间的 base 在 team_enu 中的平面位姿。"""

    stamp_ns: int
    x_enu_m: float
    y_enu_m: float
    yaw_rad: float
    z_m: float = 0.0
    quaternion_xyzw: tuple[float, float, float, float] | None = None
    velocity_enu_xy_mps: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if int(self.stamp_ns) <= 0:
            raise ValueError("POSE_STAMP_INVALID")
        if not all(math.isfinite(float(value)) for value in (self.x_enu_m, self.y_enu_m, self.yaw_rad)):
            raise ValueError("POSE_VALUE_NONFINITE")
        if self.velocity_enu_xy_mps is not None and not all(math.isfinite(float(value)) for value in self.velocity_enu_xy_mps):
            raise ValueError("POSE_VELOCITY_NONFINITE")


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _interpolate_angle(first: float, second: float, fraction: float) -> float:
    return _wrap_angle(float(first) + float(fraction) * _wrap_angle(float(second) - float(first)))


class TimestampedPoseBuffer:
    """按时间戳插值运动中的 ENU->base 关系，不允许 latest TF 回退。"""

    def __init__(self, *, maximum_gap_ns: int = 250_000_000) -> None:
        self.maximum_gap_ns = int(maximum_gap_ns)
        if self.maximum_gap_ns <= 0:
            raise ValueError("POSE_INTERPOLATION_GAP_INVALID")
        self._poses: list[PlanarPose] = []
        self._stamps: list[int] = []

    def add(self, pose: PlanarPose) -> None:
        stamp = int(pose.stamp_ns)
        index = bisect.bisect_left(self._stamps, stamp)
        if index < len(self._stamps) and self._stamps[index] == stamp:
            self._poses[index] = pose
            return
        self._stamps.insert(index, stamp)
        self._poses.insert(index, pose)

    def extend(self, poses: Iterable[PlanarPose]) -> None:
        for pose in poses:
            self.add(pose)

    def discard_before(self, stamp_ns: int) -> None:
        # 保留截点前最后一个端点，仍可为等待中的新观测提供完整插值区间。
        index = max(0, bisect.bisect_left(self._stamps, int(stamp_ns)) - 1)
        if index:
            del self._stamps[:index]
            del self._poses[:index]

    def interpolate(self, stamp_ns: int) -> PlanarPose:
        stamp = int(stamp_ns)
        if not self._poses:
            raise ValueError("POSE_BUFFER_EMPTY")
        index = bisect.bisect_left(self._stamps, stamp)
        if index < len(self._stamps) and self._stamps[index] == stamp:
            return self._poses[index]
        if index == 0 or index == len(self._poses):
            raise ValueError("POSE_TIMESTAMP_OUT_OF_RANGE")
        before = self._poses[index - 1]
        after = self._poses[index]
        gap = int(after.stamp_ns) - int(before.stamp_ns)
        if gap <= 0 or gap > self.maximum_gap_ns:
            raise ValueError("POSE_INTERPOLATION_GAP_TOO_LARGE")
        fraction = (stamp - int(before.stamp_ns)) / float(gap)
        quaternion = None
        velocity = None
        if before.velocity_enu_xy_mps is not None and after.velocity_enu_xy_mps is not None:
            velocity = tuple(first + fraction * (second - first)
                             for first, second in zip(before.velocity_enu_xy_mps, after.velocity_enu_xy_mps))
        if before.quaternion_xyzw is not None and after.quaternion_xyzw is not None:
            quaternion = tuple(Slerp([0.0, 1.0], Rotation.from_quat(
                [before.quaternion_xyzw, after.quaternion_xyzw]))([fraction]).as_quat()[0])
        return PlanarPose(
            stamp_ns=stamp,
            x_enu_m=before.x_enu_m + fraction * (after.x_enu_m - before.x_enu_m),
            y_enu_m=before.y_enu_m + fraction * (after.y_enu_m - before.y_enu_m),
            yaw_rad=_interpolate_angle(before.yaw_rad, after.yaw_rad, fraction),
            z_m=before.z_m + fraction * (after.z_m - before.z_m),
            quaternion_xyzw=quaternion,
            velocity_enu_xy_mps=velocity,
        )

    def __len__(self) -> int:
        return len(self._poses)


def sensor_to_base_xy(
    points_xy_m: Sequence[tuple[float, float]],
    transform: RigidTransform2D | RigidTransform3D,
) -> tuple[tuple[float, float], ...]:
    """用固定 sensor->base 外参转换点。"""

    result = []
    if isinstance(transform, RigidTransform3D):
        for x_sensor, y_sensor in points_xy_m:
            x_base, y_base, _ = transform.rotate_translate((float(x_sensor), float(y_sensor), 0.0))
            if not math.isfinite(x_base) or not math.isfinite(y_base):
                raise ValueError("SENSOR_POINT_TRANSFORM_NONFINITE")
            result.append((x_base, y_base))
        return tuple(result)
    cosine = math.cos(float(transform.yaw_rad))
    sine = math.sin(float(transform.yaw_rad))
    for x_sensor, y_sensor in points_xy_m:
        x_base = float(transform.translation_x_m) + cosine * float(x_sensor) - sine * float(y_sensor)
        y_base = float(transform.translation_y_m) + sine * float(x_sensor) + cosine * float(y_sensor)
        if not math.isfinite(x_base) or not math.isfinite(y_base):
            raise ValueError("SENSOR_POINT_TRANSFORM_NONFINITE")
        result.append((x_base, y_base))
    return tuple(result)


def base_to_enu_xy(point_xy_base_m: tuple[float, float], pose: PlanarPose) -> tuple[float, float]:
    """按观测时间的 base pose 将点从 base 转到 team_enu。"""

    cosine = math.cos(float(pose.yaw_rad))
    sine = math.sin(float(pose.yaw_rad))
    x_base, y_base = point_xy_base_m
    return (
        float(pose.x_enu_m) + cosine * float(x_base) - sine * float(y_base),
        float(pose.y_enu_m) + sine * float(x_base) + cosine * float(y_base),
    )


def enu_to_base_xy(point_xy_enu_m: tuple[float, float], pose: PlanarPose) -> tuple[float, float]:
    """按观测时间的 base pose 将点从 team_enu 转回 base。"""

    cosine = math.cos(float(pose.yaw_rad))
    sine = math.sin(float(pose.yaw_rad))
    x_enu, y_enu = point_xy_enu_m
    delta_x = float(x_enu) - float(pose.x_enu_m)
    delta_y = float(y_enu) - float(pose.y_enu_m)
    return (
        cosine * delta_x + sine * delta_y,
        -sine * delta_x + cosine * delta_y,
    )


__all__ = [
    "PlanarPose",
    "RigidTransform2D",
    "RigidTransform3D",
    "TimestampedPoseBuffer",
    "base_to_enu_xy",
    "enu_to_base_xy",
    "sensor_to_base_xy",
]
