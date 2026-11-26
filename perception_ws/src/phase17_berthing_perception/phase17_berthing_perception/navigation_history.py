"""感知共用生产导航的完整三维历史，按每帧观测时刻补偿运动。"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from phase16z_s3_runtime.s3_timestamped_frames import PlanarPose, TimestampedPoseBuffer
from .berthing_core import BasePose


def pose_from_odometry(message):
    if message.header.frame_id != "team_enu" or message.child_frame_id not in ("wamv/base_link", "wamv/wamv/base_link"):
        raise ValueError("NAVIGATION_FRAME_INVALID")
    p, q = message.pose.pose.position, message.pose.pose.orientation
    xyz, quaternion = (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)
    if not all(math.isfinite(v) for v in (*xyz, *quaternion)) or abs(sum(v*v for v in quaternion)-1.) > .02:
        raise ValueError("NAVIGATION_POSE_INVALID")
    rotation = Rotation.from_quat(quaternion)
    yaw = rotation.as_euler("xyz")[2]
    stamp = message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
    return PlanarPose(stamp, p.x, p.y, yaw, p.z, quaternion)


def base_pose(pose):
    roll, pitch, yaw = Rotation.from_quat(pose.quaternion_xyzw).as_euler("xyz")
    return BasePose(pose.x_enu_m, pose.y_enu_m, pose.z_m, yaw, roll, pitch)


def reproject_points(points, source_pose, anchor_pose):
    points = np.asarray(points)
    if source_pose.stamp_ns == anchor_pose.stamp_ns:
        return points
    matrix = relative_transform(source_pose, anchor_pose)
    return points@matrix[:3, :3].T+matrix[:3, 3]


def relative_transform(source_pose, anchor_pose):
    source_rotation = Rotation.from_quat(source_pose.quaternion_xyzw).as_matrix()
    anchor_rotation = Rotation.from_quat(anchor_pose.quaternion_xyzw).as_matrix()
    source_xyz = np.asarray((source_pose.x_enu_m, source_pose.y_enu_m, source_pose.z_m))
    anchor_xyz = np.asarray((anchor_pose.x_enu_m, anchor_pose.y_enu_m, anchor_pose.z_m))
    matrix = np.eye(4)
    matrix[:3, :3] = anchor_rotation.T@source_rotation
    matrix[:3, 3] = anchor_rotation.T@(source_xyz-anchor_xyz)
    return matrix


def planar_output_transform(source_pose, anchor_pose):
    """将观测时三维船体系转入锚点的水平船体系，不把倾斜面压错位置。"""
    source_rotation = Rotation.from_quat(source_pose.quaternion_xyzw).as_matrix()
    anchor_yaw = Rotation.from_euler('z', anchor_pose.yaw_rad).as_matrix()
    source_xyz = np.asarray((source_pose.x_enu_m, source_pose.y_enu_m, source_pose.z_m))
    anchor_xyz = np.asarray((anchor_pose.x_enu_m, anchor_pose.y_enu_m, anchor_pose.z_m))
    matrix = np.eye(4)
    matrix[:3, :3] = anchor_yaw.T@source_rotation
    matrix[:3, 3] = anchor_yaw.T@(source_xyz-anchor_xyz)
    return matrix


def world_xy_in_planar_frame(points_world, anchor_pose):
    """固定 ENU 几何按栅格观测时刻转入水平船体系。"""
    rotation = Rotation.from_euler('z', anchor_pose.yaw_rad).as_matrix()[:2, :2]
    return (np.asarray(points_world)-[anchor_pose.x_enu_m, anchor_pose.y_enu_m])@rotation


def snapshot_poses(buffer, clouds, anchor):
    stamps = {anchor, *(message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec for _, message in clouds)}
    return {stamp: buffer.interpolate(stamp) for stamp in stamps}
