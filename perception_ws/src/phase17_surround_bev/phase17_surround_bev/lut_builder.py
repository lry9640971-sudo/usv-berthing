"""根据标定后的六相机外参重建 10 m BEV 查找表。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import project_custom_tan as geometry
from .bev_core import CAMERA_NAMES


GRID_MAX = 10.0
GRID_RES = 0.05
GRID_SIZE = 400
FEATHER_PX = 64.0


def _grid_points():
    indices = np.arange(GRID_SIZE, dtype=np.float64)
    x_values = GRID_MAX - (indices + 0.5) * GRID_RES
    y_values = GRID_MAX - (indices + 0.5) * GRID_RES
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="ij")
    points = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros(GRID_SIZE * GRID_SIZE)))
    return x_values, y_values, x_grid, y_grid, points


def build_masked_lut(
    camera_transforms: dict[str, np.ndarray],
    geometry_contract_path: str | Path,
    static_mask_lut_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict]:
    """重建投影，并复用已审计的静态船体占地与射线遮挡代理。"""
    contract = yaml.safe_load(Path(geometry_contract_path).read_text(encoding="utf-8"))
    lens = geometry.lens_from_contract(contract)
    x_values, y_values, x_grid, y_grid, base_points = _grid_points()
    maps_u, maps_v, valid_masks = [], [], []
    incidences, ranges_out, feathers, weights = [], [], [], []
    for name in CAMERA_NAMES:
        transform = np.asarray(camera_transforms[name], dtype=np.float64)
        p_camera = geometry.transform_points(geometry.invert_transform(transform), base_points)
        projection = lens.project_sensor(p_camera)
        ranges = projection["range"]
        nonzero = ranges > 1e-12
        rays_camera = np.zeros_like(p_camera)
        rays_camera[nonzero] = p_camera[nonzero] / ranges[nonzero, None]
        rays_base = rays_camera @ transform[:3, :3].T
        incidence = np.clip(-rays_base[:, 2], 0.0, 1.0)
        edge = np.minimum.reduce(
            (
                projection["u"],
                (lens.width - 1.0) - projection["u"],
                projection["v"],
                (lens.height - 1.0) - projection["v"],
            )
        )
        feather = np.clip(edge / FEATHER_PX, 0.0, 1.0)
        valid = projection["valid"] & nonzero & (incidence > 0.0)
        raw = valid.astype(np.float64) * incidence**2 * feather**2 / np.maximum(ranges, 1.0) ** 2
        maps_u.append(projection["u"].reshape(GRID_SIZE, GRID_SIZE))
        maps_v.append(projection["v"].reshape(GRID_SIZE, GRID_SIZE))
        valid_masks.append(valid.reshape(GRID_SIZE, GRID_SIZE))
        incidences.append(incidence.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32))
        ranges_out.append(ranges.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32))
        feathers.append(feather.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32))
        weights.append(raw.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32))

    with np.load(Path(static_mask_lut_path), allow_pickle=False) as source:
        source_lut = {name: source[name] for name in source.files}
    circle = np.hypot(x_grid, y_grid) <= GRID_MAX
    self_mask = source_lut.get("bev_self_footprint_mask", source_lut["self_footprint_mask"]).astype(bool)
    ray_mask = source_lut.get(
        "bev_camera_ray_blocked_mask", np.zeros((6, GRID_SIZE, GRID_SIZE), dtype=bool)
    ).astype(bool)
    valid = np.stack(valid_masks)
    raw = np.stack(weights)
    valid &= circle[None, :, :] & ~self_mask[None, :, :] & ~ray_mask
    raw[~valid] = 0.0
    coverage = (raw > 0.0).sum(axis=0).astype(np.uint8)
    weight_sum = raw.sum(axis=0)
    normalized = np.divide(
        raw,
        weight_sum[None, :, :],
        out=np.zeros_like(raw),
        where=weight_sum[None, :, :] > 0.0,
    )
    dominant = np.argmax(raw, axis=0).astype(np.int8)
    dominant[coverage == 0] = -1
    navigable = circle & ~self_mask
    lut = {
        "camera_index": np.arange(6, dtype=np.int8),
        "x_values_m": x_values.astype(np.float32),
        "y_values_m": y_values.astype(np.float32),
        "map_u": np.stack(maps_u),
        "map_v": np.stack(maps_v),
        "valid_mask": valid,
        "viewing_incidence": np.stack(incidences),
        "source_range_m": np.stack(ranges_out),
        "boundary_feather": np.stack(feathers),
        "raw_weight": raw,
        "normalized_weight": normalized,
        "weight_sum": weight_sum.astype(np.float32),
        "coverage_count": coverage,
        "dominant_camera": dominant,
        "no_coverage_mask": coverage == 0,
        "circle_mask": circle,
        "self_body_mask": self_mask,
        "self_footprint_mask": self_mask,
        "navigable_water_mask": navigable,
        "bev_self_footprint_mask": self_mask,
        "bev_camera_ray_blocked_mask": ray_mask,
    }
    metrics = {
        "coverage_10m_circle": float(((coverage > 0) & circle).sum() / circle.sum()),
        "navigable_coverage": float(((coverage > 0) & navigable).sum() / navigable.sum()),
        "self_mask_cells": int(self_mask.sum()),
        "ray_mask_cells_per_camera": [int(value.sum()) for value in ray_mask],
        "static_ray_mask_reused_after_extrinsic_update": True,
    }
    return lut, metrics
