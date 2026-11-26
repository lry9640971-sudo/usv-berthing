"""导航与跟踪共用的坐标、消息时间和配置读取契约。"""


from __future__ import annotations
from dataclasses import dataclass
from functools import wraps
import json
import math
from pathlib import Path
import time
from typing import Any


GPS_TOPIC = "/wamv/sensors/gps/gps/fix"


IMU_TOPIC = "/wamv/sensors/imu/imu/data"


BASE_DETECTIONS_TOPIC = "/team/perception/dynamic_detections_base"


PUBLIC_FRAME = "team_enu"


BASE_FRAME = "wamv/wamv/base_link"


ODOMETRY_TOPIC = "/team/localization/odometry"


def timed_localization_callback(method):
    @wraps(method)
    def wrapped(self, message, *args, **kwargs):
        started = time.perf_counter()
        try:
            return method(self, message, *args, **kwargs)
        finally:
            metrics = getattr(self, "callback_metrics", {})
            previous = metrics.get(method.__name__, {})
            elapsed = time.perf_counter() - started
            measured_stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
            metrics[method.__name__] = {"count": previous.get("count", 0) + 1,
                "last_s": elapsed, "max_s": max(previous.get("max_s", 0.), elapsed),
                "message_age_sim_s": None if measured_stamp <= 0 else (self.get_clock().now().nanoseconds - measured_stamp) / 1e9}
            self.callback_metrics = metrics
    return wrapped


STATUS_TOPIC = "/team/localization/status"


TRACKS_TOPIC = "/team/perception/dynamic_tracks"


SYNC_TOLERANCE_NS = 100_000_000


ASYNC_FUSION_MAX_SPAN_NS = 150_000_000


DYNAMIC_SPEED_BOUND_MPS = 3.0


DYNAMIC_SPEED_MIN_MPS = 0.15


def stamp_ns(stamp: Any) -> int:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value <= 0:
        raise ValueError("ROS_HEADER_STAMP_INVALID")
    return value


def quaternion_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = float(yaw_rad) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


@dataclass(frozen=True)
class ImuState:
    stamp_ns: int
    yaw_rad: float
    frame_id: str
    transform_yaw_rad: float
    quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    angular_body_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_raw_body_xyz: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class BasePoseState:
    stamp_ns: int
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


TRACKING_STATUS_TOPIC = "/team/perception/dynamic_tracking_status"


def load_tf_proof(path: str) -> dict[str, Any]:
    if not path:
        return {}
    proof_path = Path(path)
    if not proof_path.exists():
        raise RuntimeError(f"FIXED_SENSOR_TF_PROOF_NOT_FOUND:{proof_path}")
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("FIXED_SENSOR_TF_PROOF_INVALID_JSON") from exc
    required = ("model_name", "model_source", "model_sha256", "sdf_sha256", "tf_chain", "fixed_sensor_frames")
    if not isinstance(proof, dict) or any(not proof.get(key) for key in required):
        raise RuntimeError("FIXED_SENSOR_TF_PROOF_FIELDS_MISSING")
    if proof.get("world_localization_not_inferred_from_sensor_tf") is not True:
        raise RuntimeError("FIXED_SENSOR_TF_PROOF_WORLD_LOCALIZATION_BOUNDARY_INVALID")
    if not isinstance(proof["fixed_sensor_frames"], list):
        raise RuntimeError("FIXED_SENSOR_TF_PROOF_SENSOR_FRAME_LIST_INVALID")
    return proof


def load_frame_contract(path: str) -> dict[str, Any]:
    """加载生产坐标契约；评估拟合结果不得作为生产 datum。"""
    if not path:
        raise RuntimeError("TEAM_ENU_FRAME_CONTRACT_REQUIRED")
    contract_path = Path(path)
    if not contract_path.exists():
        raise RuntimeError(f"TEAM_ENU_FRAME_CONTRACT_NOT_FOUND:{contract_path}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("TEAM_ENU_FRAME_CONTRACT_INVALID_JSON") from exc
    required = (
        "schema",
        "frame_id",
        "axes",
        "datum_source",
        "datum",
        "world_frame_orientation",
        "sdf_sha256",
    )
    if not isinstance(contract, dict) or any(not contract.get(key) for key in required):
        raise RuntimeError("TEAM_ENU_FRAME_CONTRACT_FIELDS_MISSING")
    if contract.get("schema") != "phase16z_team_enu_frame_contract_v1":
        raise RuntimeError("TEAM_ENU_FRAME_CONTRACT_SCHEMA_INVALID")
    if contract.get("frame_id") != PUBLIC_FRAME or contract.get("axes") != "x=East,y=North,z=Up":
        raise RuntimeError("TEAM_ENU_FRAME_CONTRACT_FRAME_INVALID")
    if contract.get("world_frame_orientation") != "ENU":
        raise RuntimeError("TEAM_ENU_FRAME_CONTRACT_WORLD_ORIENTATION_INVALID")
    if contract.get("evaluation_only_world_to_team_enu_fit_forbidden") is not True:
        raise RuntimeError("TEAM_ENU_FRAME_CONTRACT_EVALUATION_BOUNDARY_INVALID")
    datum = contract.get("datum")
    if not isinstance(datum, dict) or any(key not in datum for key in ("latitude_deg", "longitude_deg", "altitude_m")):
        raise RuntimeError("TEAM_ENU_FRAME_CONTRACT_DATUM_INVALID")
    values = tuple(float(datum[key]) for key in ("latitude_deg", "longitude_deg", "altitude_m"))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("TEAM_ENU_FRAME_CONTRACT_DATUM_NONFINITE")
    return contract


def stamp_to_msg(value_ns: int) -> Any:
    from builtin_interfaces.msg import Time as RosTime

    result = RosTime()
    result.sec = int(value_ns // 1_000_000_000)
    result.nanosec = int(value_ns % 1_000_000_000)
    return result
