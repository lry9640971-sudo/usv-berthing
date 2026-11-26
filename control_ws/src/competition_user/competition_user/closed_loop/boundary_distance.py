"""精确边界距离：格子只索引可能最近的线段，不栅格化船体或净空。"""

import numpy as np
import shapely


class BoundaryDistanceIndex:
    def __init__(self, hull, cell_size=.2):
        self.hull, self.cell_size = hull, cell_size
        segments = []
        for ring in shapely.get_parts(hull.boundary):
            coordinates = np.asarray(ring.coords)
            segments.append(np.stack((coordinates[:-1], coordinates[1:]), axis=1))
        segments = np.concatenate(segments)
        self.starts, self.edges = segments[:, 0], segments[:, 1]-segments[:, 0]
        length_sq = np.sum(self.edges*self.edges, axis=1)
        self.length_sq = np.where(length_sq > 0., length_sq, 1.)
        self.origin = -hull.radius-2.
        self.width = int(np.ceil(-2*self.origin/cell_size))
        y, x = np.indices((self.width, self.width))
        centers = self.origin+cell_size*(np.column_stack((x.ravel(), y.ravel()))+.5)
        counts, indices = [], []
        diameter = np.sqrt(2.)*cell_size
        for begin in range(0, len(centers), 128):
            delta = centers[begin:begin+128, None, :]-self.starts[None]
            ratio = np.clip(np.sum(delta*self.edges[None], axis=-1)/self.length_sq[None], 0., 1.)
            distances = np.linalg.norm(delta-ratio[..., None]*self.edges[None], axis=-1)
            # 距离函数为1-Lipschitz。比格心最近距离大两个格半径的段，
            # 在该格任何点都不可能成为最近段；只是精确索引，不栅格化距离。
            eligible = distances <= distances.min(axis=1)[:, None]+diameter+1e-12
            counts.extend(eligible.sum(axis=1).tolist())
            indices.append(np.nonzero(eligible)[1])
        self.counts = np.asarray(counts, dtype=int)
        self.offsets = np.concatenate(([0], np.cumsum(self.counts)))
        self.indices = np.concatenate(indices)

    def signed_distance(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not len(points):
            return np.empty(0)
        finite = np.isfinite(points).all(axis=1)
        cells = np.zeros((len(points), 2), dtype=int)
        cells[finite] = np.floor((points[finite]-self.origin)/self.cell_size).astype(int)
        eligible = finite & (cells >= 0).all(axis=1) & (cells < self.width).all(axis=1)
        result = np.full(len(points), np.nan)
        active = np.flatnonzero(eligible)
        if len(active):
            cell_ids = cells[active, 1]*self.width+cells[active, 0]
            counts = self.counts[cell_ids]
            positions = np.concatenate(([0], np.cumsum(counts)))
            distances = np.empty(len(active))
            begin = 0
            while begin < len(active):
                # 按点与候选段的配对数分块，避免大轨迹池构造数百万行临时数组。
                # 每点仍计算缓存中的全部原线段，分块只限制工作内存。
                end = max(begin+1, int(np.searchsorted(positions, positions[begin]+65536, side="right"))-1)
                end = min(end, len(active))
                block_counts = counts[begin:end]
                block_positions = positions[begin:end+1]-positions[begin]
                owner = np.repeat(np.arange(begin, end), block_counts)
                segment_ids = self.indices[np.repeat(self.offsets[cell_ids[begin:end]], block_counts)
                    +np.arange(block_positions[-1])-np.repeat(block_positions[:-1], block_counts)]
                delta = points[active[owner]]-self.starts[segment_ids]
                ratio = np.clip(np.sum(delta*self.edges[segment_ids], axis=1)/self.length_sq[segment_ids], 0., 1.)
                residual = delta-ratio[:, None]*self.edges[segment_ids]
                squared = np.sum(residual*residual, axis=1)
                distances[begin:end] = np.sqrt(np.minimum.reduceat(squared, block_positions[:-1]))
                begin = end
            inside = shapely.contains_xy(self.hull.geometry, points[active, 0], points[active, 1])
            result[active] = np.where(inside, -distances, distances)
        # 远场沿用原完整空间索引；缓存范围没有改变船体或障碍查询范围。
        other = np.flatnonzero(~eligible)
        if len(other):
            indices, distances = self.hull.boundary_index.query_nearest(shapely.points(points[other]),
                return_distance=True, all_matches=False)
            values = np.full(len(other), np.nan)
            values[indices[0]] = distances
            inside = shapely.contains_xy(self.hull.geometry, points[other, 0], points[other, 1])
            result[other] = np.where(inside, -values, values)
        return result
