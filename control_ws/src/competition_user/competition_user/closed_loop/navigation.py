"""观测地图和粗路径；粗路径只作引导，实际推力轨迹仍检查完整船体。"""

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from .geometry import transform_xy


class ObservationMap:
    def __init__(self, resolution_m=.2):
        self.resolution = resolution_m
        self.occupied = {}

    def update(self, grid, origin_pose, ship_pose, cell_resolution):
        # 膨胀格 50 不是新的物理回波；空间余量由实际船体和格子尺寸统一计算。
        occupied = np.argwhere(grid == 100)
        free = np.argwhere(grid == 0)

        def world_points(indices):
            xy = (indices[:, ::-1] + .5) * cell_resolution
            return transform_xy(transform_xy(xy, origin_pose), ship_pose)

        # 清除证据必须覆盖原障碍所在格，不能按相邻FREE所在的粗桶清空。
        if self.occupied:
            keys = list(self.occupied)
            old = np.asarray([self.occupied[key] for key in keys])
            def inverse(points, pose):
                x, y, yaw = pose
                c, s = math.cos(yaw), math.sin(yaw)
                return transform_xy(points, (-c*x-s*y, s*x-c*y, -yaw))
            local = inverse(inverse(old, ship_pose), origin_pose)
            cells = np.floor(local/cell_resolution).astype(int)
            valid = ((cells[:, 0] >= 0) & (cells[:, 0] < grid.shape[1])
                     & (cells[:, 1] >= 0) & (cells[:, 1] < grid.shape[0]))
            for index in np.flatnonzero(valid):
                x, y = cells[index]
                if grid[y, x] == 0:
                    del self.occupied[keys[index]]
        points = world_points(occupied)
        keys, indices = np.unique(np.floor(points / self.resolution).astype(np.int64), axis=0, return_index=True)
        for key, point in zip(keys, points[indices]):
            self.occupied[tuple(map(int, key))] = tuple(map(float, point))
        return points

    def nearby(self, position, maximum_range=45.):
        points = np.asarray(list(self.occupied.values()), dtype=float).reshape(-1, 2)
        return points[np.linalg.norm(points - position, axis=1) <= maximum_range]


class RoutePlanner:
    def __init__(self, hull, radius_m=38., resolution_m=.8):
        self.hull, self.radius, self.resolution = hull, radius_m, resolution_m
        self.path = []
        self.last_plan = -math.inf
        self.last_goal = None
        self.status = "NOT_PLANNED"

    def plan(self, position, goal, obstacles, stamp):
        if self.path and stamp - self.last_plan < 1. and self.last_goal is not None and np.linalg.norm(np.asarray(goal)-self.last_goal) < .4:
            return self.path
        self.last_plan, self.last_goal = stamp, np.asarray(goal)
        origin = np.floor((np.asarray(position) - self.radius) / self.resolution) * self.resolution
        size = int(math.ceil(2 * self.radius / self.resolution)) + 1
        blocked = np.zeros((size, size), dtype=bool)
        points = np.asarray(obstacles).reshape(-1, 2)
        cells = np.floor((points - origin) / self.resolution).astype(int)
        cells = cells[(cells >= 0).all(axis=1) & (cells < size).all(axis=1)]
        if len(cells):
            blocked[cells[:, 1], cells[:, 0]] = True
        clearance = distance_transform_edt(~blocked) * self.resolution if blocked.any() else np.full(blocked.shape, np.inf)
        # 导航宽度依据本船横向尺寸；后续有向船体检查约束转弯和前后伸出。
        half_width = max(abs(self.hull.bounds[1]), abs(self.hull.bounds[3]))
        blocked |= clearance < half_width + .25 + self.resolution / 2
        start = tuple(np.floor((np.asarray(position) - origin) / self.resolution).astype(int))
        delta = np.asarray(goal) - position
        distance = np.linalg.norm(delta)
        local_goal = np.asarray(position) + delta * min(1., (self.radius - 3.) / max(distance, 1e-6))
        target = tuple(np.floor((local_goal - origin) / self.resolution).astype(int))
        coordinates = np.indices(blocked.shape).transpose(1, 2, 0)[:, :, ::-1]
        radial = np.linalg.norm(origin + (coordinates + .5) * self.resolution - position, axis=2)
        blocked |= radial > self.radius
        if blocked[start[1], start[0]]:
            # 粗网格不足以表示当前船体姿态时不伪造原地路径；有向推力规划仍检查观测。
            self.path, self.status = [], "COARSE_START_BLOCKED"
            return self.path
        heap = [(0., start)]
        previous, distances = {}, {start: 0.}
        best, best_distance = start, math.inf
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        visited = set()
        while heap:
            _, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            h = math.hypot(current[0] - target[0], current[1] - target[1])
            if not blocked[current[1], current[0]] and h < best_distance:
                best, best_distance = current, h
            if current == target:
                best = current
                break
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < size and 0 <= ny < size) or blocked[ny, nx]:
                    continue
                if dx and dy and (blocked[current[1], nx] or blocked[ny, current[0]]):
                    continue
                step = math.hypot(dx, dy) + .08 / max(.2, clearance[ny, nx])
                candidate = distances[current] + step
                if candidate < distances.get((nx, ny), math.inf):
                    distances[(nx, ny)] = candidate
                    previous[(nx, ny)] = current
                    heuristic = math.hypot(nx - target[0], ny - target[1])
                    heapq.heappush(heap, (candidate + heuristic, (nx, ny)))
        cells_path = [best]
        while cells_path[-1] in previous:
            cells_path.append(previous[cells_path[-1]])
        cells_path.reverse()
        self.path = [tuple(origin + (np.asarray(cell) + .5) * self.resolution) for cell in cells_path]
        self.status = "ROUTE_AVAILABLE" if len(self.path) > 1 else "NO_COARSE_ROUTE"
        if len(self.path) == 1:
            self.path = []
        return self.path

    def lookahead(self, position, distance_m=4.):
        if not self.path:
            return None
        values = np.asarray(self.path)
        nearest = int(np.linalg.norm(values - position, axis=1).argmin())
        travel = 0.
        for i in range(nearest + 1, len(values)):
            travel += float(np.linalg.norm(values[i] - values[i - 1]))
            if travel >= distance_m:
                return tuple(values[i])
        return self.path[-1]
