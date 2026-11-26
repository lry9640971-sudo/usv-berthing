"""完整真实船体与观测障碍的有向距离，安全底线不依赖粗规划外接圆。"""

from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import ConvexHull, cKDTree
import yaml
from .boundary_distance import BoundaryDistanceIndex


def transform_xy(points, pose):
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(points) @ np.asarray(((c, s), (-s, c))) + (x, y)


def moving_clearances(poses, hull, obstacles, dt, times_s=None):
    """按每个预测姿态检查真实部件并集，目标半径及速度不确定度保持原值。"""
    poses = np.asarray(poses, dtype=float)
    if not obstacles:
        return np.full(poses.shape[:-1], np.inf)
    values = np.asarray([(item.x, item.y, item.vx, item.vy, item.radius_m, item.velocity_std_mps)
                         for item in obstacles])
    times = np.arange(poses.shape[-2])*dt if times_s is None else np.asarray(times_s, dtype=float)
    centers = values[None, :, :2]+times[:, None, None]*values[None, :, 2:4]
    delta = centers-poses[..., None, :2]
    c, s = np.cos(poses[..., 2])[..., None], np.sin(poses[..., 2])[..., None]
    local = np.stack((c*delta[..., 0]+s*delta[..., 1], -s*delta[..., 0]+c*delta[..., 1]), axis=-1)
    distance = hull.signed_distance(local).reshape(local.shape[:-1])
    radii = values[None, :, 4]+times[:, None]*values[None, :, 5]
    return (distance-radii).min(axis=-1)


def swept_interval_bounds(poses, hull, obstacles, times, static, dynamic):
    """由船体平移、转角及目标运动给出两姿态之间的净空下界。"""
    poses, times = np.asarray(poses), np.asarray(times)
    static, dynamic = np.asarray(static), np.asarray(dynamic)
    if poses.ndim == 2:
        poses = np.stack((poses[:-1], poses[1:]), axis=1)
        times = np.stack((times[:-1], times[1:]), axis=1)
        static = np.stack((static[:-1], static[1:]), axis=1)
        dynamic = np.stack((dynamic[:-1], dynamic[1:]), axis=1)
    turn = poses[:, 1, 2] - poses[:, 0, 2]
    turn = np.arctan2(np.sin(turn), np.cos(turn))
    travel = np.linalg.norm(poses[:, 1, :2] - poses[:, 0, :2], axis=1) + hull.radius * np.abs(turn)
    target_rate = max((np.hypot(item.vx, item.vy) + item.velocity_std_mps for item in obstacles), default=0.)
    # 插值段任一点到较近端点的船体运动不超过总运动的一半；半径增长也计入。
    return (static.min(axis=1) - travel / 2.,
            dynamic.min(axis=1) - (travel + target_rate * (times[:, 1] - times[:, 0])) / 2.)


def certify_swept_poses(poses, field, obstacles, times, deadline=None):
    """自适应精查预测折线姿态间隙；未证明安全的段不能标记SAFE。"""
    import time

    poses, times = np.asarray(poses, dtype=float), np.asarray(times, dtype=float)
    static = field.clearances(poses)
    dynamic = moving_clearances(poses, field.hull, obstacles, 0., times_s=times)
    sampled_static, sampled_dynamic = float(static.min()), float(dynamic.min())
    evaluated, rounds = len(poses), 0
    poses = np.stack((poses[:-1], poses[1:]), axis=1)
    times = np.stack((times[:-1], times[1:]), axis=1)
    static = np.stack((static[:-1], static[1:]), axis=1)
    dynamic = np.stack((dynamic[:-1], dynamic[1:]), axis=1)
    static_bound = dynamic_bound = np.inf
    while True:
        if min(sampled_static, sampled_dynamic) < 0.:
            return False, sampled_static, sampled_dynamic, {"reason": "SWEPT_COLLISION", "poses": evaluated, "refinements": rounds}
        lower_static, lower_dynamic = swept_interval_bounds(poses, field.hull, obstacles, times, static, dynamic)
        certified = (lower_static >= 0.) & (lower_dynamic >= 0.)
        if certified.any():
            static_bound = min(static_bound, float(lower_static[certified].min()))
            dynamic_bound = min(dynamic_bound, float(lower_dynamic[certified].min()))
        if certified.all():
            return True, sampled_static, sampled_dynamic, {"reason": "SWEPT_CERTIFIED", "poses": evaluated,
                "refinements": rounds, "static_lower_bound_m": static_bound, "dynamic_lower_bound_m": dynamic_bound}
        if deadline is not None and time.perf_counter() >= deadline:
            return False, sampled_static, sampled_dynamic, {"reason": "CERTIFICATION_BUDGET_REACHED", "poses": evaluated, "refinements": rounds}
        unresolved = np.flatnonzero(~certified)
        first, last = poses[unresolved, 0], poses[unresolved, 1]
        midpoint = (first + last) / 2.
        turn = np.arctan2(np.sin(last[:, 2] - first[:, 2]), np.cos(last[:, 2] - first[:, 2]))
        midpoint[:, 2] = first[:, 2] + turn / 2.
        first_time, last_time = times[unresolved, 0], times[unresolved, 1]
        middle_time = (first_time + last_time) / 2.
        if np.any((middle_time == first_time) | (middle_time == last_time)):
            return False, sampled_static, sampled_dynamic, {"reason": "UNRESOLVED_CONTACT_BOUNDARY", "poses": evaluated, "refinements": rounds}
        middle_static = field.clearances(midpoint)
        middle_dynamic = moving_clearances(midpoint, field.hull, obstacles, 0., times_s=middle_time)
        sampled_static = min(sampled_static, float(middle_static.min()))
        sampled_dynamic = min(sampled_dynamic, float(middle_dynamic.min()))
        evaluated += len(midpoint)
        rounds += 1
        # 已证实的段无需重算；只将未决段拆成两个，保持各自真实预测时刻。
        poses = np.concatenate((np.stack((first, midpoint), axis=1), np.stack((midpoint, last), axis=1)))
        times = np.concatenate((np.stack((first_time, middle_time), axis=1), np.stack((middle_time, last_time), axis=1)))
        static = np.concatenate((np.stack((static[unresolved, 0], middle_static), axis=1),
                                 np.stack((middle_static, static[unresolved, 1]), axis=1)))
        dynamic = np.concatenate((np.stack((dynamic[unresolved, 0], middle_dynamic), axis=1),
                                  np.stack((middle_dynamic, dynamic[unresolved, 1]), axis=1)))


class Hull:
    def __init__(self, vertices, components=None):
        vertices = np.asarray(vertices, dtype=float)
        convex = ConvexHull(vertices)
        self.vertices = vertices[convex.vertices]
        self.normals, self.offsets = convex.equations[:, :2], convex.equations[:, 2]
        self.edges = np.roll(self.vertices, -1, axis=0) - self.vertices
        self.edge_length_sq = np.sum(self.edges * self.edges, axis=1)
        self.radius = float(np.linalg.norm(vertices, axis=1).max())
        self.bounds = (*vertices.min(axis=0), *vertices.max(axis=0))
        self.components = (self.vertices,) if components is None else tuple(components)
        self.geometry = shapely.union_all([shapely.Polygon(part) for part in self.components])
        self.boundary = self.geometry.boundary
        # 所有原并集边界线段进入空间索引，包括孔洞；精确最近距离无需逐点扫全边界。
        segments = []
        for ring in shapely.get_parts(self.boundary):
            coordinates = np.asarray(ring.coords)
            segments.append(np.stack((coordinates[:-1], coordinates[1:]), axis=1))
        self.boundary_index = shapely.STRtree(shapely.linestrings(np.concatenate(segments)))
        self.convex_geometry = shapely.Polygon(self.vertices)
        shapely.prepare(self.geometry)
        shapely.prepare(self.convex_geometry)
        self.distance_index = BoundaryDistanceIndex(self)

    @classmethod
    def load(cls, path):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        parts = [np.asarray(part["polygon_xy"], dtype=float) for part in data["components_polygon_xy"]]
        # 所有实际部件做几何并集；不把浮筒和发动机之间的空隙填实。
        return cls(np.concatenate(parts), parts)

    def signed_distance(self, local_points):
        return self.distance_index.signed_distance(local_points)

    def convex_lower_distance(self, points):
        # 单位外法向的最大平面距离是部件并集距离的下界；外部角点也只作筛选。
        # 最终距离仍由 signed_distance 精查，避免每次筛选都构造大批GEOS点。
        points = np.asarray(points)
        result = np.empty(len(points))
        # 完整凸包平面分块计算，不生成所有障碍点乘全部法向的大矩阵。
        for begin in range(0, len(points), 2048):
            end = begin+2048
            result[begin:end] = (points[begin:end]@self.normals.T+self.offsets).max(axis=1)
        return result


class CollisionField:
    def __init__(self, points, hull, point_radius_m=0.):
        self.points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not np.isfinite(self.points).all():
            raise ValueError("障碍坐标包含无效值")
        self.tree = cKDTree(self.points) if len(self.points) else None
        self.hull, self.point_radius = hull, point_radius_m

    def clearances(self, poses):
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        if self.tree is None:
            return np.full(len(poses), np.inf)
        center_distances, nearest = self.tree.query(poses[:, :2])
        delta = self.points[nearest]-poses[:, :2]
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        local = np.column_stack((c*delta[:, 0]+s*delta[:, 1], -s*delta[:, 0]+c*delta[:, 1]))
        upper = self.hull.signed_distance(local)
        # 完整船体之外两米仍参与舒适代价；更远点只需确定不可能碰撞。
        # 最近中心点提供真实距离上界。任何可能更近的点都在 R+max(上界,0) 内，
        # 内部点包含在 R 内；查询半径仍不超过原范围，远场返回约定保持不变。
        original_radius = self.hull.radius + self.point_radius + 2.
        radius = np.minimum(original_radius, self.hull.radius+np.maximum(upper, 0.)+1e-9)
        nearby = self.tree.query_ball_point(poses[:, :2], radius, return_sorted=False)
        result = center_distances - self.hull.radius - self.point_radius
        counts = np.asarray([len(indices) for indices in nearby], dtype=int)
        if counts.sum():
            owner = np.repeat(np.arange(len(poses)), counts)
            indices = np.concatenate([row for row in nearby if len(row)])
            delta = self.points[indices]-poses[owner, :2]
            c, s = np.cos(poses[owner, 2]), np.sin(poses[owner, 2])
            local = np.column_stack((c*delta[:, 0]+s*delta[:, 1], -s*delta[:, 0]+c*delta[:, 1]))
            # 凸包距离只是并集距离的下界；仅精查仍可能优于已知上界的点。
            # 每个姿态的最终值仍是原始部件并集的精确最小值，不以凸包阻止动作。
            left, bottom, right, top = self.hull.bounds
            rectangle_lower = np.maximum.reduce((left-local[:, 0], local[:, 0]-right,
                                                 bottom-local[:, 1], local[:, 1]-top))
            # 先用外包矩形的四个平面排除不可能胜过上界的点，再计算完整凸包下界。
            # 两者都只是原并集距离下界，剩余点仍走完整精确边界。
            possible = rectangle_lower <= upper[owner]+1e-9
            possible[possible] = (self.hull.convex_lower_distance(local[possible])
                                  <= upper[owner[possible]]+1e-9)
            distances = self.hull.signed_distance(local[possible])-self.point_radius
            result[counts > 0] = np.inf
            np.minimum.at(result, owner[possible], distances)
        return result
