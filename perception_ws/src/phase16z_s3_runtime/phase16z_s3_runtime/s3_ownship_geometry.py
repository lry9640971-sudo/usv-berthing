"""模型自身三维部件过滤；不使用场景位置和外部障碍标签。"""

from pathlib import Path
import numpy as np
import yaml


class OwnshipGeometry:
    def __init__(self, components):
        self.components = tuple(components)
        if not self.components:
            raise ValueError("OWN_GEOMETRY_EMPTY")
        self.bounds = np.asarray((
            np.min([bounds[0] - margin for bounds, _, margin in self.components], axis=0),
            np.max([bounds[1] + margin for bounds, _, margin in self.components], axis=0)))

    def contains(self, points_xyz):
        points = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
        inside = np.zeros(len(points), dtype=bool)
        finite = np.isfinite(points).all(axis=1)
        # 船体外的远处点只做一次包围盒判断，不参与逐部件半空间运算。
        selected = np.flatnonzero(finite & np.all((points >= self.bounds[0]) & (points <= self.bounds[1]), axis=1))
        local_points = points[selected]
        local_inside = np.zeros(len(selected), dtype=bool)
        for bounds, planes, margin in self.components:
            candidates = ~local_inside & np.all(
                (local_points >= bounds[0] - margin) & (local_points <= bounds[1] + margin), axis=1)
            indices = np.flatnonzero(candidates)
            if len(indices):
                signed = local_points[indices] @ planes[:, :3].T + planes[:, 3]
                local_inside[indices] = np.all(signed <= margin + 1e-8, axis=1)
        inside[selected] = local_inside
        return inside


def load_ownship_geometry(path):
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "phase17/self_visual_geometry_v2":
        raise ValueError("OWN_GEOMETRY_SCHEMA_INVALID")
    rows = document.get("components", [])
    if len(rows) != int(document.get("component_count", -1)):
        raise ValueError("OWN_GEOMETRY_COMPONENT_COUNT_INVALID")
    components = []
    for row in rows:
        bounds = np.asarray(row["bounds_xyz"], dtype=float)
        planes = np.asarray(row["halfspaces_xyz"], dtype=float)
        margin = float(row["margin_m"])
        if (bounds.shape != (2, 3) or planes.ndim != 2 or planes.shape[1] != 4
                or len(planes) < 4 or not np.isfinite(bounds).all()
                or not np.isfinite(planes).all() or np.any(bounds[1] < bounds[0])
                or not 0. <= margin <= .1
                or not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-6)):
            raise ValueError("OWN_GEOMETRY_COMPONENT_INVALID")
        components.append((bounds, planes, margin))
    return OwnshipGeometry(components)
