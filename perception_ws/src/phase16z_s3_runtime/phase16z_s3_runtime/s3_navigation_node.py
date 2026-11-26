"""独立 GPS/IMU 导航进程；不执行候选、关联或船体语义计算。"""

from __future__ import annotations
from collections import deque
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import String
from scipy.spatial.transform import Rotation
import tf2_ros
import yaml

from .s3_localization_math import ConstantAccelerationKalman2D, ConstantVelocityKalman2D, WGS84LocalENU, quaternion_to_yaw, unwrap_angle
from .s3_odometry_contract import localization_fresh
from .s3_orientation_rate import OrientationRate
from .s3_runtime_contracts import (
    GPS_TOPIC, IMU_TOPIC, BASE_FRAME, PUBLIC_FRAME, ODOMETRY_TOPIC, STATUS_TOPIC,
    SYNC_TOLERANCE_NS, BasePoseState, ImuState, load_frame_contract, load_tf_proof,
    quaternion_from_yaw, stamp_ns, stamp_to_msg, timed_localization_callback,
)


class S3NavigationNode(Node):
    """短导航回调保持独立，重跟踪负载不会阻塞里程计发布。"""

    def __init__(self):
        super().__init__("phase16z_navigation")
        config = Path(get_package_share_directory("phase16z_s3_runtime")) / "config"
        for name, default in (("receipt_path", ""), ("gps_topic", GPS_TOPIC), ("imu_topic", IMU_TOPIC),
                              ("tf_proof_path", str(config / "fixed_sensor_tf_proof.json")),
                              ("frame_contract_path", str(config / "team_enu_frame_contract.json")),
                              ("gps_velocity_topic", ""), ("gps_position_variance_m2", 4.), ("fuse_gps_velocity", True),
                              ("motion_model", "constant_velocity"),
                              ("use_model_sensor_mounts", False)):
            self.declare_parameter(name, default)
        receipt = str(self.get_parameter("receipt_path").value).strip()
        self.receipt_path = Path(receipt) if receipt else None
        self.gps_topic = str(self.get_parameter("gps_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.tf_proof = load_tf_proof(str(self.get_parameter("tf_proof_path").value))
        self.frame_contract = load_frame_contract(str(self.get_parameter("frame_contract_path").value))
        self.fixed_sensor_frames = set(self.tf_proof.get("fixed_sensor_frames", []))
        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=False)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.use_model_sensor_mounts = bool(self.get_parameter("use_model_sensor_mounts").value)
        self.model_sensor_mounts = {}
        self.mount_tf_diagnostics = {}
        self.mount_tf_checked_stamp = {}
        if self.use_model_sensor_mounts:
            mounts = yaml.safe_load((config / "model_sensor_mounts.yaml").read_text(encoding="utf-8"))
            if mounts["base_frame"] != BASE_FRAME:
                raise ValueError("MODEL_SENSOR_MOUNT_FRAME_INVALID")
            self.model_sensor_mounts = mounts["sensor_frames"]
        output_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.odom_publisher = self.create_publisher(Odometry, ODOMETRY_TOPIC, output_qos)
        self.status_publisher = self.create_publisher(String, STATUS_TOPIC, output_qos)
        self.gps_velocity_topic = str(self.get_parameter("gps_velocity_topic").value)
        self.fuse_gps_velocity = bool(self.get_parameter("fuse_gps_velocity").value)
        self.gps_position_variance = float(self.get_parameter("gps_position_variance_m2").value)
        if self.gps_position_variance <= 0:
            raise ValueError("GPS_POSITION_VARIANCE_INVALID")
        self.gps_velocity_rejection_count = 0
        self._gps_fix_waiting = {}
        self._gps_velocity_waiting = {}
        self.gps_velocity_pair_count = 0
        self.gps_subscription = self.create_subscription(NavSatFix, self.gps_topic, self.gps_pair_callback, qos_profile_sensor_data)
        if self.gps_velocity_topic:
            self.gps_velocity_subscription = self.create_subscription(TwistStamped, self.gps_velocity_topic,
                                                                      self.gps_velocity_callback, 20)
        self.imu_subscription = self.create_subscription(Imu, self.imu_topic, self.imu_callback, qos_profile_sensor_data)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.started_monotonic = time.monotonic()
        self.callback_metrics = {}
        for name in ("gps_message_count", "gps_valid_count", "gps_eligible_count", "gps_rejected_count",
                     "gps_outlier_count", "imu_message_count", "imu_valid_count", "imu_rejected_count",
                     "localization_output_count", "tf_gps_success_count", "tf_imu_success_count",
                     "tf_exact_success_count", "tf_exact_miss_count", "tf_latest_rigid_fallback_count", "tf_failure_count"):
            setattr(self, name, 0)
        self.tf_gps_translation = self.tf_gps_rotation_yaw_rad = None
        self.tf_imu_translation = self.tf_imu_rotation_yaw_rad = None
        self.gps_frames, self.imu_frames = set(), set()
        self.datum = None
        self.motion_model = str(self.get_parameter("motion_model").value)
        self.filter = {"constant_velocity": ConstantVelocityKalman2D,
                       "constant_acceleration": ConstantAccelerationKalman2D}[self.motion_model]()
        self.imu_history = deque(maxlen=1000)
        # 原零限位安装的姿态抖动会被短窗口放大为假角速度，按真实回放采用.2s运动窗口。
        self.orientation_rate = OrientationRate(.2)
        self.angular_rate_source = "BOOTSTRAP_RAW_GYRO"
        self.angular_rate_raw_residual = None
        self.attitude_source = "BOOTSTRAP_RAW_ORIENTATION"
        self.lever_projection_variance_xy = (0., 0.)
        self.last_imu_yaw_rad = self.last_gps_stamp_ns = self.last_localization_stamp_ns = None
        self._last_localization_wall = None
        self.last_status_reason = "WAITING_FOR_FIRST_VALID_GPS_AND_IMU"

    def _lookup_base_sensor_tf(self, source_frame: str, stamp: int) -> Any:
        if not source_frame:
            raise ValueError("SENSOR_FRAME_MISSING")
        if self.use_model_sensor_mounts:
            # 比赛原船模声明零运动范围安装；不能把实时TF的数值抖动再求导成船体运动。
            mount = self.model_sensor_mounts.get(source_frame)
            if mount is None:
                raise ValueError("MODEL_SENSOR_MOUNT_UNDECLARED")
            result = TransformStamped()
            result.header.frame_id, result.child_frame_id = BASE_FRAME, source_frame
            result.header.stamp = stamp_to_msg(stamp)
            translation, rotation = result.transform.translation, result.transform.rotation
            translation.x, translation.y, translation.z = map(float, mount["translation_xyz_m"])
            rotation.x, rotation.y, rotation.z, rotation.w = map(float, mount["quaternion_xyzw"])
            if stamp-self.mount_tf_checked_stamp.get(source_frame, -1_000_000_000) >= 500_000_000:
                self.mount_tf_checked_stamp[source_frame] = stamp
                try:
                    live = self.tf_buffer.lookup_transform(BASE_FRAME, source_frame, Time(nanoseconds=int(stamp)))
                except Exception as exc:
                    self.mount_tf_diagnostics[source_frame] = {"status": "TF_DIAGNOSTIC_UNAVAILABLE", "reason": str(exc)}
                else:
                    q, p = live.transform.rotation, live.transform.translation
                    delta_rotation = Rotation.from_quat(mount["quaternion_xyzw"]).inv()*Rotation.from_quat([q.x, q.y, q.z, q.w])
                    self.mount_tf_diagnostics[source_frame] = {"status": "TF_DIAGNOSTIC_AVAILABLE",
                        "rotation_delta_rad": float(delta_rotation.magnitude()),
                        "translation_delta_m": float(np.linalg.norm(np.asarray([p.x, p.y, p.z])-mount["translation_xyz_m"]))}
            return result
        try:
            result = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                source_frame,
                Time(nanoseconds=int(stamp)),
            )
            self.tf_exact_success_count += 1
            return result
        except Exception as exact_exc:
            self.tf_exact_miss_count += 1
            # 仅对模型证明的固定 sensor->base 外参使用 latest；运动中的
            # ENU->base 仍始终按观测时间戳从 pose_history 查询。
            if source_frame not in self.fixed_sensor_frames:
                self.tf_failure_count += 1
                raise ValueError("SENSOR_TO_BASE_TF_AT_OBSERVATION_TIME_UNAVAILABLE") from exact_exc
            try:
                rigid_result = self.tf_buffer.lookup_transform(BASE_FRAME, source_frame, Time())
            except Exception as latest_exc:
                self.tf_failure_count += 1
                raise ValueError("SENSOR_TO_BASE_TF_AT_OBSERVATION_TIME_UNAVAILABLE") from latest_exc
            if (
                str(rigid_result.header.frame_id).strip() != BASE_FRAME
                or str(rigid_result.child_frame_id).strip() != source_frame
            ):
                self.tf_failure_count += 1
                raise ValueError("SENSOR_TO_BASE_TF_CHAIN_MISMATCH") from exact_exc
            self.tf_latest_rigid_fallback_count += 1
            return rigid_result

    @staticmethod
    def _transform_yaw(transform: Any) -> float:
        rotation = transform.transform.rotation
        return quaternion_to_yaw(rotation.x, rotation.y, rotation.z, rotation.w)

    @timed_localization_callback
    def imu_callback(self, message: Imu) -> None:
        self.imu_message_count += 1
        frame_id = str(message.header.frame_id).strip()
        if frame_id:
            self.imu_frames.add(frame_id)
        try:
            stamp = stamp_ns(message.header.stamp)
            raw_yaw = quaternion_to_yaw(
                message.orientation.x,
                message.orientation.y,
                message.orientation.z,
                message.orientation.w,
            )
            transform = self._lookup_base_sensor_tf(frame_id, stamp)
            transform_yaw = self._transform_yaw(transform)
            # 完整旋转复合：ENU<-base = ENU<-IMU 乘以 base<-IMU 的逆。
            sensor_q = message.orientation
            fixed_q = transform.transform.rotation
            base_rotation = Rotation.from_quat([sensor_q.x, sensor_q.y, sensor_q.z, sensor_q.w]) * Rotation.from_quat(
                [fixed_q.x, fixed_q.y, fixed_q.z, fixed_q.w]).inv()
            base_quaternion = tuple(float(v) for v in base_rotation.as_quat())
            base_yaw = unwrap_angle(self.last_imu_yaw_rad, quaternion_to_yaw(*base_quaternion))
            self.last_imu_yaw_rad = base_yaw
            angular = Rotation.from_quat([fixed_q.x, fixed_q.y, fixed_q.z, fixed_q.w]).apply(
                [message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z])
            raw_angular = tuple(float(value) for value in angular)
            rate = self.orientation_rate.observe(stamp, base_quaternion)
            if rate is not None:
                self.angular_rate_raw_residual = tuple(float(value) for value in angular-np.asarray(rate))
                angular = rate
                self.angular_rate_source = "ORIENTATION_BODY_MOTION"
            self.imu_history.append(ImuState(stamp, base_yaw, frame_id, transform_yaw, base_quaternion,
                                             tuple(float(v) for v in angular), raw_angular))
            self.imu_valid_count += 1
            self.tf_imu_success_count += 1
            if self.tf_imu_translation is None:
                self.tf_imu_translation = (
                    float(transform.transform.translation.x),
                    float(transform.transform.translation.y),
                    float(transform.transform.translation.z),
                )
                self.tf_imu_rotation_yaw_rad = transform_yaw
        except (TypeError, ValueError) as exc:
            self.imu_rejected_count += 1
            self.last_status_reason = str(exc)

    def _nearest_imu(self, stamp: int) -> ImuState | None:
        if not self.imu_history:
            return None
        candidate = min(self.imu_history, key=lambda item: abs(item.stamp_ns - stamp))
        return candidate if abs(candidate.stamp_ns - stamp) <= SYNC_TOLERANCE_NS else None

    def _gps_covariance_xy(self, message: NavSatFix) -> tuple[float, float]:
        covariance = tuple(float(value) for value in message.position_covariance)
        east = covariance[0] if len(covariance) >= 1 and math.isfinite(covariance[0]) and covariance[0] > 1e-6 else self.gps_position_variance
        north = covariance[4] if len(covariance) >= 5 and math.isfinite(covariance[4]) and covariance[4] > 1e-6 else self.gps_position_variance
        return (min(east, 100.0), min(north, 100.0))

    def gps_pair_callback(self, message: NavSatFix) -> None:
        if not self.gps_velocity_topic or not self.fuse_gps_velocity:
            self.gps_callback(message)
            return
        stamp = stamp_ns(message.header.stamp)
        velocity = self._gps_velocity_waiting.pop(stamp, None)
        if velocity is None:
            self._gps_fix_waiting[stamp] = message
            self._prune_gps_pairing(stamp)
        else:
            self.gps_callback(message, velocity)

    def _prune_gps_pairing(self, latest_stamp):
        # 保留一秒以容纳两条真实传输路径；不拿旧速度配新位置。
        for pending in (self._gps_fix_waiting, self._gps_velocity_waiting):
            for stamp in list(pending):
                if latest_stamp - stamp > 1_000_000_000:
                    del pending[stamp]

    def gps_velocity_callback(self, message: TwistStamped) -> None:
        values = (message.twist.linear.x, message.twist.linear.y, message.twist.linear.z)
        if message.header.frame_id != PUBLIC_FRAME or not all(math.isfinite(v) for v in values):
            self.last_status_reason = "GPS_VELOCITY_FRAME_OR_VALUES_INVALID"
            return
        stamp = stamp_ns(message.header.stamp)
        if not self.fuse_gps_velocity:
            self._gps_velocity_waiting[stamp] = values
            self._prune_gps_pairing(stamp)
            return
        fix = self._gps_fix_waiting.pop(stamp, None)
        if fix is None:
            self._gps_velocity_waiting[stamp] = values
            self._prune_gps_pairing(stamp)
        else:
            self.gps_callback(fix, values)

    @timed_localization_callback
    def gps_callback(self, message: NavSatFix, velocity_enu=None) -> None:
        self.gps_message_count += 1
        frame_id = str(message.header.frame_id).strip()
        if frame_id:
            self.gps_frames.add(frame_id)
        try:
            stamp = stamp_ns(message.header.stamp)
            if self.last_gps_stamp_ns is not None and stamp <= self.last_gps_stamp_ns:
                raise ValueError("GPS_STAMP_NOT_INCREASING")
            latitude = float(message.latitude)
            longitude = float(message.longitude)
            altitude = float(message.altitude)
            if not all(math.isfinite(value) for value in (latitude, longitude, altitude)):
                raise ValueError("GPS_FIX_NONFINITE")
            if self.datum is None:
                # datum 来自实际运行 SDF 的 spherical_coordinates，不使用首个 GPS fix。
                datum = self.frame_contract["datum"]
                scene_lat = float(datum["latitude_deg"])
                scene_lon = float(datum["longitude_deg"])
                scene_alt = float(datum["altitude_m"])
                self.datum = WGS84LocalENU.from_fix(scene_lat, scene_lon, scene_alt)
                self.get_logger().info(
                    f"ENU datum initialized from SDF reference: "
                    f"lat={scene_lat:.6f}, lon={scene_lon:.6f}, alt={scene_alt:.1f}"
                )
            gps_enu = self.datum.project(latitude, longitude, altitude)
            imu = self._nearest_imu(stamp)
            if imu is None:
                raise ValueError("GPS_IMU_TIME_SYNC_UNAVAILABLE")
            fit = self.orientation_rate.pose_at(imu.stamp_ns, imu.quaternion_xyzw)
            if fit is not None:
                quaternion, _ = fit
                yaw = unwrap_angle(imu.yaw_rad, quaternion_to_yaw(*quaternion))
                imu = replace(imu, quaternion_xyzw=quaternion, yaw_rad=yaw)
                self.attitude_source = "CAUSAL_QUADRATIC_ORIENTATION"
            else:
                self.attitude_source = "BOOTSTRAP_RAW_ORIENTATION"
            transform = self._lookup_base_sensor_tf(frame_id, stamp)
            self.tf_gps_success_count += 1
            if self.tf_gps_translation is None:
                self.tf_gps_translation = (
                    float(transform.transform.translation.x),
                    float(transform.transform.translation.y),
                    float(transform.transform.translation.z),
                )
                self.tf_gps_rotation_yaw_rad = self._transform_yaw(transform)
            lever = transform.transform.translation
            rotation = Rotation.from_quat(imu.quaternion_xyzw)
            lever_xyz = [lever.x, lever.y, lever.z]
            lever_enu = rotation.apply(lever_xyz)
            base_x, base_y, base_z = np.asarray(gps_enu) - lever_enu
            covariance_xy = np.asarray(self._gps_covariance_xy(message))
            self.lever_projection_variance_xy = (0., 0.)
            if fit is not None:
                x, y, z = lever_xyz
                skew = np.asarray(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
                # GPS本身的方差不能代表IMU姿态校正后的船体中心位置方差。
                jacobian = -(rotation.as_matrix()@skew)[:2]
                projected = np.diag(jacobian@fit[1]@jacobian.T)
                self.lever_projection_variance_xy = tuple(float(value) for value in projected)
                covariance_xy += projected
            update = self.filter.update(stamp, (base_x, base_y), covariance_xy)
            self.last_gps_stamp_ns = stamp
            self.gps_valid_count += 1
            if not update.accepted:
                self.gps_outlier_count += 1
                self.last_status_reason = "GPS_JUMP_REJECTED"
                return
            if velocity_enu is not None and self.fuse_gps_velocity:
                rotation = Rotation.from_quat(imu.quaternion_xyzw)
                # 瞬时NavSat速度只使用同刻原始角速度，不能混用窗口平均率。
                angular_enu = rotation.apply(imu.angular_raw_body_xyz or imu.angular_body_xyz)
                base_velocity = np.asarray(velocity_enu) - np.cross(angular_enu, lever_enu)
                # 原始 NavSat 不含速度噪声；校正误差主要来自实际 0.009 rad/s 的 IMU 噪声。
                variance = max(1e-6, .009**2 * float(np.dot(lever_enu, lever_enu)))
                if self.filter.update_velocity(base_velocity[:2], variance):
                    self.gps_velocity_pair_count += 1
                else:
                    self.gps_velocity_rejection_count += 1
            self.gps_eligible_count += 1
            self.last_localization_stamp_ns = stamp
            self._publish_localization(stamp, float(base_z), imu.yaw_rad, imu.quaternion_xyzw)
        except (TypeError, ValueError) as exc:
            self.gps_rejected_count += 1
            self.last_status_reason = str(exc)

    def _publish_localization(self, stamp: int, z_m: float, yaw_rad: float, quaternion_xyzw=None) -> None:
        state = self.filter.state
        covariance = self.filter.covariance
        pose = BasePoseState(stamp, float(state[0]), float(state[1]), float(z_m), float(yaw_rad),
                             quaternion_xyzw or quaternion_from_yaw(yaw_rad))
        odometry = Odometry()
        odometry.header.stamp = stamp_to_msg(stamp)
        odometry.header.frame_id = PUBLIC_FRAME
        odometry.child_frame_id = BASE_FRAME
        odometry.pose.pose.position.x = pose.x_m
        odometry.pose.pose.position.y = pose.y_m
        odometry.pose.pose.position.z = pose.z_m
        qx, qy, qz, qw = pose.quaternion_xyzw
        odometry.pose.pose.orientation.x = qx
        odometry.pose.pose.orientation.y = qy
        odometry.pose.pose.orientation.z = qz
        odometry.pose.pose.orientation.w = qw
        odometry.pose.covariance[0] = float(covariance[0, 0])
        odometry.pose.covariance[1] = float(covariance[0, 1])
        odometry.pose.covariance[6] = float(covariance[1, 0])
        odometry.pose.covariance[7] = float(covariance[1, 1])
        from .s3_odometry_contract import body_velocity_covariance
        body_velocity, body_covariance = body_velocity_covariance(state[2:4], covariance[2:4, 2:4], pose.quaternion_xyzw)
        odometry.twist.twist.linear.x, odometry.twist.twist.linear.y, odometry.twist.twist.linear.z = map(float, body_velocity)
        for row in range(3):
            for column in range(3):
                odometry.twist.covariance[row * 6 + column] = float(body_covariance[row, column])
        imu = min(self.imu_history, key=lambda item: abs(item.stamp_ns - stamp), default=None)
        if imu is not None and abs(imu.stamp_ns - stamp) <= 250_000_000:
            odometry.twist.twist.angular.x, odometry.twist.twist.angular.y, odometry.twist.twist.angular.z = imu.angular_body_xyz
            for index in (21, 28, 35):
                odometry.twist.covariance[index] = .01
        else:
            for index in (21, 28, 35):
                odometry.twist.covariance[index] = 100.
        self.odom_publisher.publish(odometry)
        self._last_localization_wall = time.monotonic()

        transform = TransformStamped()
        transform.header = odometry.header
        transform.header.frame_id = PUBLIC_FRAME
        transform.child_frame_id = BASE_FRAME
        transform.transform.translation.x = pose.x_m
        transform.transform.translation.y = pose.y_m
        transform.transform.translation.z = pose.z_m
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)
        self.localization_output_count += 1
        self.last_status_reason = "LOCALIZATION_READY"

    def publish_status(self):
        now = self.get_clock().now().nanoseconds
        imu_stamp = self.imu_history[-1].stamp_ns if self.imu_history else None
        wall_age = None if self._last_localization_wall is None else time.monotonic() - self._last_localization_wall
        ready = localization_fresh(self.last_gps_stamp_ns, imu_stamp, now, wall_age)
        payload = {
            "schema": "phase16z_navigation_status_v2", "status": "READY" if ready else "NOT_READY",
            "reason": self.last_status_reason if ready else "LOCALIZATION_INPUT_STALE_OR_MISSING",
            "public_frame": PUBLIC_FRAME, "base_frame": BASE_FRAME,
            "gps_age_ms": None if self.last_gps_stamp_ns is None else (now - self.last_gps_stamp_ns) / 1e6,
            "imu_age_ms": None if imu_stamp is None else (now - imu_stamp) / 1e6,
            "gps_message_count": self.gps_message_count, "imu_message_count": self.imu_message_count,
            "localization_output_count": self.localization_output_count, "callback_metrics": self.callback_metrics,
            "gps_velocity_topic": self.gps_velocity_topic, "gps_velocity_pairs": self.gps_velocity_pair_count,
            "gps_velocity_fused": self.fuse_gps_velocity, "gps_position_variance_m2": self.gps_position_variance,
            "motion_model": self.motion_model,
            "estimated_acceleration_enu_mps2": self.filter.state[4:].tolist() if self.filter.state.size == 6 else None,
            "sensor_mount_source": "MODEL_ZERO_RANGE_MOUNT" if self.use_model_sensor_mounts else "TF_AT_MEASUREMENT",
            "mount_live_tf_deviation": self.mount_tf_diagnostics,
            "gps_velocity_rejections": self.gps_velocity_rejection_count,
            "angular_motion_window_s": self.orientation_rate.window_ns/1e9,
            "gps_pair_pending": {"fix": len(self._gps_fix_waiting), "velocity": len(self._gps_velocity_waiting)},
            "angular_rate_source": self.angular_rate_source,
            "angular_rate_raw_residual_xyz_rad_s": self.angular_rate_raw_residual,
            "angular_rate_variance_xyz": self.orientation_rate.variance.tolist(),
            "attitude_source": self.attitude_source,
            "lever_projection_variance_xy_m2": self.lever_projection_variance_xy,
        }
        self.status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def write_receipt(self):
        if self.receipt_path is None:
            return
        receipt = {
            "schema": "phase16z_navigation_receipt_v2", "frame_contract": self.frame_contract,
            "gps_messages": self.gps_message_count, "gps_rejected": self.gps_rejected_count,
            "imu_messages": self.imu_message_count, "imu_rejected": self.imu_rejected_count,
            "localization_outputs": self.localization_output_count, "callback_metrics": self.callback_metrics,
            "input_topics": [self.gps_topic, self.imu_topic] + ([self.gps_velocity_topic] if self.gps_velocity_topic else []),
            "gps_velocity_pairs": self.gps_velocity_pair_count,
            "motion_model": self.motion_model,
            "angular_rate_source": self.angular_rate_source,
            "odometry_topic": ODOMETRY_TOPIC,
            "elapsed_wall_s": time.monotonic() - self.started_monotonic,
        }
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = S3NavigationNode()
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        if node is not None:
            node.write_receipt()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
