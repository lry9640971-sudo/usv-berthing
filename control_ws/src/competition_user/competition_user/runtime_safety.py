"""输入时效、测量时刻坐标和真实障碍边界的公共内核。"""

import math

import numpy as np
from scipy.spatial import cKDTree


def input_is_fresh(receipt, now_sim, now_wall, maximum_sim_age, maximum_wall_age):
    """测量和接收两种年龄都必须合格，不能用接收时刻重写旧测量。"""
    if receipt is None:
        return False
    stamp, wall = receipt
    return (-0.05 <= now_sim - stamp <= maximum_sim_age
            and 0.0 <= now_wall - wall <= maximum_wall_age)


def pose_at_stamp(history, stamp, maximum_gap=0.30, extrapolation_limit=0.10):
    """插值测量时刻的二维船体位姿，禁止用当前位姿处理旧点云。"""
    if not history:
        return None
    times = np.asarray([row[0] for row in history])
    index = int(np.searchsorted(times, stamp))
    if index < len(history) and abs(times[index] - stamp) < 1e-6:
        return tuple(history[index][1:])
    if index == 0 or index == len(history):
        row = history[0] if index == 0 else history[-1]
        return tuple(row[1:]) if abs(stamp - row[0]) <= extrapolation_limit else None
    left, right = history[index - 1], history[index]
    dt = right[0] - left[0]
    if not 0.0 < dt <= maximum_gap:
        return None
    fraction = (stamp - left[0]) / dt
    yaw_delta = math.atan2(math.sin(right[3] - left[3]), math.cos(right[3] - left[3]))
    return (left[1] + fraction * (right[1] - left[1]),
            left[2] + fraction * (right[2] - left[2]), left[3] + fraction * yaw_delta)


def transform_points(points, pose):
    """批量将船体平面点转换到固定 ENU。"""
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return values @ np.asarray(((c, s), (-s, c))) + np.asarray((x, y))


def occupied_grid_points(data, width, height, resolution, origin_pose, ship_pose):
    """保留真实命中格；不把整块中心冒充命中位置。"""
    values = np.asarray(data, dtype=np.int8)
    if width <= 0 or height <= 0 or values.size != width * height or resolution <= 0:
        raise ValueError("OCCUPANCY_GRID_SHAPE_INVALID")
    rows, columns = np.where(values.reshape(height, width) >= 50)
    local = np.column_stack(((columns + .5) * resolution, (rows + .5) * resolution))
    return transform_points(transform_points(local, origin_pose), ship_pose)


def filter_known_dock_structure(points, template, dock_pose, tolerance=.45):
    """仅豁免新鲜配准模板中的结构点，通道内独立障碍始终保留。"""
    if not points or dock_pose is None or template is None or not len(template):
        return list(points)
    x, y, yaw = dock_pose
    c, s = math.cos(yaw), math.sin(yaw)
    local = (np.asarray(points, dtype=float) - (x, y)) @ np.asarray(((c, -s), (s, c)))
    distances, _ = cKDTree(np.asarray(template)).query(local)
    return [point for point, distance in zip(points, distances) if distance > tolerance]


def unique_obstacle_points(points, resolution=.4):
    """只选择实际观测点作空间摘要，完整点另用于最终安全检查。"""
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return []
    _, indices = np.unique(np.floor(values / resolution).astype(np.int64), axis=0, return_index=True)
    return [tuple(row) for row in values[np.sort(indices)]]


def footprint_clearances(points, pose, footprint=(-2.5, 2.5, -1.23, 1.23), margin=.20):
    """批量返回每个点到真实朝向船体边界的有符号间距。"""
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(values):
        return np.empty(0)
    xmin, xmax, ymin, ymax = footprint
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    local = (values - (x, y)) @ np.asarray(((c, -s), (s, c)))
    dx = np.maximum(np.maximum(xmin - margin - local[:, 0], local[:, 0] - xmax - margin), 0.)
    dy = np.maximum(np.maximum(ymin - margin - local[:, 1], local[:, 1] - ymax - margin), 0.)
    outside = np.hypot(dx, dy)
    inside = np.minimum.reduce((local[:, 0] - xmin + margin, xmax + margin - local[:, 0],
                                local[:, 1] - ymin + margin, ymax + margin - local[:, 1]))
    return np.where((dx == 0.) & (dy == 0.), -inside, outside)


def trajectory_minimum_clearance(points, trajectory, footprint=(-2.5, 2.5, -1.23, 1.23), margin=.20):
    """按有方向的船体矩形检查轨迹，避免外接圆误封可通过的窄通道。"""
    if not len(points):
        return math.inf
    minimum = math.inf
    for pose in trajectory:
        minimum = min(minimum, float(footprint_clearances(points, pose, footprint, margin).min()))
    return minimum


def command_trajectory(pose, velocity, command, horizon=2., dt=.10):
    """保留当前惯性和侧向漂移，不能把停车指令当成瞬时静止。"""
    x, y, yaw = pose
    v, lateral, w = velocity
    target_v, target_w = command
    trajectory = [(x, y, yaw)]
    for _ in range(int(math.ceil(horizon / dt))):
        v += (target_v - v) * min(1., dt / .50)
        w += (target_w - w) * min(1., dt / .75)
        x += (v * math.cos(yaw) - lateral * math.sin(yaw)) * dt
        y += (v * math.sin(yaw) + lateral * math.cos(yaw)) * dt
        yaw += w * dt
        lateral *= math.exp(-dt / 1.5)
        trajectory.append((x, y, yaw))
    return trajectory


def dynamic_trajectory_clearance(trajectory, positions, velocities, dt=.10, hard_floor=None):
    if hard_floor is None:
        from .vessel_geometry import DYNAMIC_COLLISION_DISTANCE
        hard_floor = DYNAMIC_COLLISION_DISTANCE
    """逐时刻比较移动目标，避免只检查首尾预测点漏掉中间相遇。"""
    if not positions:
        return math.inf, math.inf
    values = np.asarray(positions, dtype=float).reshape(-1, 2)
    rates = np.asarray([velocities.get(tuple(p), (0., 0.)) for p in positions])
    initial = float(np.linalg.norm(values - np.asarray(trajectory[0][:2]), axis=1).min()) - hard_floor
    minimum = initial
    for index, pose in enumerate(trajectory[1:], 1):
        moved = values + rates * index * dt
        minimum = min(minimum, float(np.linalg.norm(moved - pose[:2], axis=1).min()) - hard_floor)
    return initial, minimum
