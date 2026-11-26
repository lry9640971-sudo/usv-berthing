"""里程计消息的速度坐标合同，遵循 child_frame_id 约定。"""

import numpy as np
from scipy.spatial.transform import Rotation


def body_velocity_covariance(world_velocity_xy, world_covariance_xy, quaternion_xyzw):
    """平面 ENU 速度及其不确定度旋转到完整船体坐标。"""
    inverse = Rotation.from_quat(quaternion_xyzw).as_matrix().T
    velocity = inverse @ np.asarray((*world_velocity_xy, 0.))
    covariance = np.zeros((3, 3))
    covariance[:2, :2] = np.asarray(world_covariance_xy)
    covariance[2, 2] = 100.  # 平面滤波器没有估计垂向速度，不声称零方差。
    return velocity, inverse @ covariance @ inverse.T


def localization_fresh(gps_stamp_ns, imu_stamp_ns, now_ns, wall_age_s):
    if gps_stamp_ns is None or imu_stamp_ns is None or now_ns <= 0 or wall_age_s is None:
        return False
    return (-50_000_000 <= now_ns - gps_stamp_ns <= 750_000_000
            and -50_000_000 <= now_ns - imu_stamp_ns <= 500_000_000
            and 0. <= wall_age_s <= 3.)
