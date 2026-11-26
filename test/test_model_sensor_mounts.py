"""比赛固定安装契约：实时TF抖动只诊断，不改变姿态运动或等待无关速度流。"""

from pathlib import Path
from types import SimpleNamespace

from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest
import rclpy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, NavSatFix

from phase16z_s3_runtime import s3_navigation_node as module
from phase16z_s3_runtime.s3_runtime_contracts import BASE_FRAME, stamp_to_msg


def test_competition_mount_profile_preserves_true_rotation_despite_live_tf_jitter(monkeypatch):
    package = Path(__file__).resolve().parents[1]/"perception_ws/src/phase16z_s3_runtime"
    monkeypatch.setattr(module, "get_package_share_directory", lambda _: str(package))
    rclpy.init(args=["--ros-args", "--params-file", str(package/"config/sensor_navigation.yaml"),
                    "-p", "gps_velocity_topic:=/diagnostic_velocity"])
    node = None
    try:
        node = module.S3NavigationNode()
        outputs = []
        node.odom_publisher = SimpleNamespace(publish=outputs.append)
        node.tf_broadcaster = SimpleNamespace(sendTransform=lambda _: None)

        def jittered_tf(*args):
            transform = TransformStamped()
            transform.transform.rotation.z = float(np.sin(.02/2))
            transform.transform.rotation.w = float(np.cos(.02/2))
            return transform
        node.tf_buffer.lookup_transform = jittered_tf
        for index in range(76):
            t = index*.01
            message = Imu()
            message.header.frame_id = "wamv/wamv/imu_wamv_link/imu_wamv_sensor"
            message.header.stamp = stamp_to_msg(round((1.+t)*1e9))
            q = Rotation.from_euler("z", 3.+3.6*t).as_quat()
            message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = map(float, q)
            node.imu_callback(message)
        assert node.imu_history[-1].angular_body_xyz == pytest.approx((0., 0., 3.6), abs=1e-10)
        imu_frame = "wamv/wamv/imu_wamv_link/imu_wamv_sensor"
        assert node.mount_tf_diagnostics[imu_frame]["rotation_delta_rad"] == pytest.approx(.02)
        gps_frame = "wamv/wamv/gps_wamv_link/navsat"
        transform = node._lookup_base_sensor_tf(gps_frame, 1_750_000_000)
        p = transform.transform.translation
        assert (p.x, p.y, p.z) == (-.85, 0., 1.3)
        assert transform.header.frame_id == BASE_FRAME
        fix = NavSatFix()
        fix.header.frame_id, fix.header.stamp = gps_frame, stamp_to_msg(1_750_000_000)
        datum = node.frame_contract["datum"]
        fix.latitude, fix.longitude, fix.altitude = [datum[key] for key in ("latitude_deg", "longitude_deg", "altitude_m")]
        node.gps_pair_callback(fix)
        assert len(outputs) == 1
        assert not node._gps_fix_waiting
        assert node.fuse_gps_velocity is False
        actual = outputs[0].pose.pose.orientation
        assert np.asarray([actual.x, actual.y, actual.z, actual.w]) == pytest.approx(q)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
