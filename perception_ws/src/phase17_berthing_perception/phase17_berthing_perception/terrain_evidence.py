"""从重力方向、有限回波和可观测坡面提取地形证据，不读取场景地图。"""

import numpy as np


def terrain_evidence(points_base, elevation, normal_up, coherent, eligible, mean_water_elevation_m=0.):
    points = np.asarray(points_base, dtype=float)
    elevation = np.asarray(elevation, dtype=float)
    # 海平面基准来自 ENU/水动力契约，不能把最大的低平陆地拟合成水面。
    seeds = eligible & (normal_up >= .95) & (np.abs(elevation - mean_water_elevation_m) <= .08)
    water, land = np.zeros(len(points), dtype=bool), np.zeros(len(points), dtype=bool)
    water_height = float(mean_water_elevation_m)
    observed_water = False
    if seeds.sum() >= 100 and np.min(np.ptp(points[seeds, :2], axis=0)) >= 2.:
        water_height += float(np.clip(np.median(elevation[seeds]) - water_height, -.04, .04))
        observed_water = True
    result = {"water_elevation_m": water_height, "observed_water_reference": observed_water,
              "terrain_points": 0, "terrain_edge_points": 0, "accepted_slope_patches": 0}
    water = eligible & (elevation <= water_height + .10) & (normal_up >= .8)
    land = eligible & coherent & (normal_up >= .82) & (elevation >= water_height + .14) & (elevation <= water_height + 1.5)
    indices = np.flatnonzero(land)
    result["terrain_points"] = int(len(indices))
    if len(indices) < 30:
        return water, land, np.empty((0, 2)), result
    # 仅用真实观测拟合局部坡面；降采样防止高密度单一路占满计算预算。
    _, chosen = np.unique(np.floor(points[indices, :2] / .30).astype(np.int64), axis=0, return_index=True)
    sample, heights = points[indices[chosen], :2], elevation[indices[chosen]]
    tiles, edges = np.floor(sample / 4.).astype(np.int64), []
    for tile in np.unique(tiles, axis=0):
        nearby = (np.max(np.abs(sample - (tile + .5) * 4.), axis=1) <= 3.) & (heights <= water_height + .9)
        if nearby.sum() < 18:
            continue
        xy, z = sample[nearby], heights[nearby]
        mean = np.mean(xy, axis=0)
        design = np.column_stack((xy - mean, np.ones(len(xy))))
        coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
        inliers = np.abs(z - design @ coefficients) <= .06
        if inliers.sum() < 18 or float(np.mean(inliers)) < .8:
            continue
        coefficients, *_ = np.linalg.lstsq(design[inliers], z[inliers], rcond=None)
        gradient = coefficients[:2]
        magnitude = float(np.linalg.norm(gradient))
        if not .025 <= magnitude <= .35 or np.min(np.linalg.eigvalsh(np.cov(xy[inliers].T))) < .10:
            continue
        core = np.all(tiles == tile, axis=1) & (heights <= water_height + .50)
        if not core.any():
            continue
        original = sample[core]
        fitted_height = (original - mean) @ gradient + coefficients[2]
        # 最多外推四米；无坡度或不一致区域不外推，不能凭空补整幅岸线。
        distance = (fitted_height - water_height - .02) / magnitude
        acceptable = (distance > 0.) & (distance <= 4.)
        if acceptable.any():
            edges.append(original[acceptable] - distance[acceptable, None] * gradient / magnitude)
            result["accepted_slope_patches"] += 1
    boundary = np.concatenate(edges) if edges else np.empty((0, 2))
    result["terrain_edge_points"] = int(len(boundary))
    return water, land, boundary, result
