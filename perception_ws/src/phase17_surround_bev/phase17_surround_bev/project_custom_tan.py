#!/usr/bin/env python3
"""Gazebo custom tan 鱼眼投影与冻结标称几何的纯数学实现。"""

from __future__ import annotations

import hashlib
import json
import math
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


CAMERA_NAMES = ("front", "back", "left", "left2", "right", "right2")


def _module_cli() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gazebo custom tan 投影数学库；实际验证请使用 "
            "run_nominal_projection_sanity.py"
        )
    )
    parser.parse_args()
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """固定轴 RPY：Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def pose_to_matrix(xyz: list[float], rpy: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_matrix(*[float(value) for value in rpy])
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def matrix_to_list(matrix: np.ndarray, digits: int = 12) -> list[list[float]]:
    return np.round(matrix.astype(float), digits).tolist()


@dataclass(frozen=True)
class CustomTanLens:
    width: int
    height: int
    horizontal_fov_rad: float
    c1: float
    c2: float
    c3: float
    configured_f: float
    scale_to_hfov: bool
    cutoff_angle_rad: float

    @property
    def aspect(self) -> float:
        return float(self.width) / float(self.height)

    @property
    def effective_f(self) -> float:
        if self.scale_to_hfov:
            argument = (
                (self.horizontal_fov_rad / 2.0) / self.c2 + self.c3
            )
            return 1.0 / (self.c1 * math.tan(argument))
        return self.configured_f

    @property
    def center(self) -> tuple[float, float]:
        return self.width / 2.0, self.height / 2.0

    def radial(self, theta: np.ndarray) -> np.ndarray:
        return (
            self.c1
            * self.effective_f
            * np.tan(theta / self.c2 + self.c3)
        )

    def project_sensor(
        self,
        points_sensor: np.ndarray,
        *,
        render_up_sign: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """将 ROS camera sensor frame 点投影到连续图像坐标。

        sensor: +X 前、+Y 左、+Z 上。
        render: +X 右、+Y 上、+Z 光轴。
        """
        points = np.asarray(points_sensor, dtype=float)
        right = -points[:, 1]
        up = render_up_sign * points[:, 2]
        forward = points[:, 0]
        rho = np.hypot(right, up)
        theta = np.arctan2(rho, forward)
        radial = self.radial(theta)
        x_norm = np.zeros_like(radial)
        y_norm = np.zeros_like(radial)
        nonzero = rho > 1e-15
        x_norm[nonzero] = radial[nonzero] * right[nonzero] / rho[nonzero]
        y_norm[nonzero] = radial[nonzero] * up[nonzero] / rho[nonzero]
        u = (x_norm / 2.0 + 0.5) * self.width
        v = (1.0 - (y_norm * self.aspect / 2.0 + 0.5)) * self.height
        ranges = np.linalg.norm(points, axis=1)
        finite = np.isfinite(points).all(axis=1)
        valid = (
            finite
            & np.isfinite(u)
            & np.isfinite(v)
            & (theta <= self.cutoff_angle_rad + 1e-12)
            & (u >= 0.0)
            & (u < float(self.width))
            & (v >= 0.0)
            & (v < float(self.height))
        )
        return {
            "u": u,
            "v": v,
            "theta": theta,
            "range": ranges,
            "valid": valid,
            "render_xyz": np.column_stack((right, up, forward)),
        }

    def inverse_pixel_rays_sensor(
        self, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """连续图像坐标反投影为 sensor frame 单位射线。"""
        u_array = np.asarray(u, dtype=float)
        v_array = np.asarray(v, dtype=float)
        x_norm = 2.0 * u_array / self.width - 1.0
        y_screen = 1.0 - 2.0 * v_array / self.height
        y_norm = y_screen / self.aspect
        radial = np.hypot(x_norm, y_norm)
        argument = np.arctan(radial / (self.c1 * self.effective_f))
        theta = self.c2 * (argument - self.c3)
        phi = np.arctan2(y_norm, x_norm)
        sin_theta = np.sin(theta)
        right = sin_theta * np.cos(phi)
        up = sin_theta * np.sin(phi)
        forward = np.cos(theta)
        rays_sensor = np.column_stack((forward, -right, up))
        rays_sensor /= np.linalg.norm(rays_sensor, axis=1, keepdims=True)
        valid = (
            np.isfinite(rays_sensor).all(axis=1)
            & (theta >= 0.0)
            & (theta <= self.cutoff_angle_rad + 1e-12)
            & (u_array >= 0.0)
            & (u_array <= float(self.width))
            & (v_array >= 0.0)
            & (v_array <= float(self.height))
        )
        return rays_sensor, valid

    def valid_pixel_mask(self) -> np.ndarray:
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        u = xx.reshape(-1).astype(float) + 0.5
        v = yy.reshape(-1).astype(float) + 0.5
        _, valid = self.inverse_pixel_rays_sensor(u, v)
        return valid.reshape(self.height, self.width)


def _required_text(element: ET.Element | None, field: str) -> str:
    if element is None or element.text is None:
        raise ValueError(f"缺少版本化相机字段：{field}")
    return element.text.strip()


def _sdformat_default(schema_root: ET.Element, element_name: str) -> str:
    found = schema_root.find(f".//element[@name='{element_name}']")
    if found is None or "default" not in found.attrib:
        raise ValueError(f"SDFormat schema 缺少默认值：{element_name}")
    return found.attrib["default"]


def build_nominal_contract(
    source_contract_path: Path,
    expanded_urdf_path: Path,
    sdformat_camera_schema_path: Path,
) -> dict[str, Any]:
    source = json.loads(source_contract_path.read_text(encoding="utf-8"))
    if source.get("commit") != "75babf08680736cb932b8e078fdeae1531a10134":
        raise ValueError("source_contract commit 与冻结版本不一致")
    if source.get("tree") != "9952e074f1113f4091f32d69e4ad9309b1b17423":
        raise ValueError("source_contract tree 与冻结版本不一致")

    urdf_root = ET.parse(expanded_urdf_path).getroot()
    sdf_schema_root = ET.parse(sdformat_camera_schema_path).getroot()
    camera_elements: dict[str, ET.Element] = {}
    for sensor in urdf_root.findall(".//sensor"):
        name = sensor.attrib.get("name", "")
        if name in {f"{item}_camera_sensor" for item in CAMERA_NAMES}:
            camera_elements[name] = sensor
    if len(camera_elements) != 6:
        raise ValueError(f"展开 URDF 只找到 {len(camera_elements)} 个目标相机")

    lens_values: list[dict[str, Any]] = []
    for name in CAMERA_NAMES:
        sensor = camera_elements[f"{name}_camera_sensor"]
        camera = sensor.find("camera")
        if camera is None:
            raise ValueError(f"{name} 缺少 camera 元素")
        lens = camera.find("lens")
        if lens is None:
            raise ValueError(f"{name} 缺少 lens 元素")
        custom = lens.find("custom_function")
        if custom is None:
            raise ValueError(f"{name} 缺少 custom_function")
        image = camera.find("image")
        if image is None:
            raise ValueError(f"{name} 缺少 image 元素")
        c3_node = custom.find("c3")
        cutoff_node = lens.find("cutoff_angle")
        item = {
            "sensor": name,
            "sensor_type": sensor.attrib.get("type"),
            "width": int(_required_text(image.find("width"), "width")),
            "height": int(_required_text(image.find("height"), "height")),
            "horizontal_fov_rad": float(
                _required_text(camera.find("horizontal_fov"), "horizontal_fov")
            ),
            "c1": float(_required_text(custom.find("c1"), "c1")),
            "c2": float(_required_text(custom.find("c2"), "c2")),
            "c3": float(
                c3_node.text.strip()
                if c3_node is not None and c3_node.text
                else _sdformat_default(sdf_schema_root, "c3")
            ),
            "configured_f": float(_required_text(custom.find("f"), "f")),
            "fun": _required_text(custom.find("fun"), "fun"),
            "scale_to_hfov": _required_text(
                lens.find("scale_to_hfov"), "scale_to_hfov"
            ).lower()
            == "true",
            "cutoff_angle_rad": float(
                cutoff_node.text.strip()
                if cutoff_node is not None and cutoff_node.text
                else _sdformat_default(sdf_schema_root, "cutoff_angle")
            ),
        }
        lens_values.append(item)
    first_lens = {key: value for key, value in lens_values[0].items() if key != "sensor"}
    if any(
        {key: value for key, value in item.items() if key != "sensor"}
        != first_lens
        for item in lens_values[1:]
    ):
        raise ValueError("六路相机的展开 lens 参数不一致")
    if first_lens["sensor_type"] != "wide_angle_camera":
        raise ValueError("目标相机不是 wide_angle_camera")
    if first_lens["fun"] != "tan" or not first_lens["scale_to_hfov"]:
        raise ValueError("本实现仅接受冻结的 custom tan + scale_to_hfov=true")

    sensors = {item["name"]: item for item in source["perception_sensors"]}
    transforms: dict[str, Any] = {}
    for name in CAMERA_NAMES:
        camera_key = f"{name}_camera_sensor"
        lidar_key = f"{name}_lidar_sensor"
        if camera_key not in sensors or lidar_key not in sensors:
            raise ValueError(f"source_contract 缺少配对：{name}")
        camera_pose = sensors[camera_key]["nominal_pose_from_base_link"]
        lidar_pose = sensors[lidar_key]["nominal_pose_from_base_link"]
        t_base_camera = pose_to_matrix(
            camera_pose["xyz_m"], camera_pose["rpy_rad"]
        )
        t_base_lidar = pose_to_matrix(
            lidar_pose["xyz_m"], lidar_pose["rpy_rad"]
        )
        t_camera_lidar = invert_transform(t_base_camera) @ t_base_lidar
        transforms[name] = {
            "T_base_camera": matrix_to_list(t_base_camera),
            "T_base_lidar": matrix_to_list(t_base_lidar),
            "T_camera_lidar": matrix_to_list(t_camera_lidar),
            "inverse_product_max_abs_error": float(
                np.max(
                    np.abs(
                        t_camera_lidar
                        @ invert_transform(t_camera_lidar)
                        - np.eye(4)
                    )
                )
            ),
        }

    forward = sensors.get("forward_lidar_sensor")
    if forward is None or "lidar" not in forward:
        raise ValueError("source_contract 缺少 forward EMX")
    forward_pose = forward["nominal_pose_from_base_link"]
    t_base_forward = pose_to_matrix(
        forward_pose["xyz_m"], forward_pose["rpy_rad"]
    )

    lens = CustomTanLens(
        width=first_lens["width"],
        height=first_lens["height"],
        horizontal_fov_rad=first_lens["horizontal_fov_rad"],
        c1=first_lens["c1"],
        c2=first_lens["c2"],
        c3=first_lens["c3"],
        configured_f=first_lens["configured_f"],
        scale_to_hfov=first_lens["scale_to_hfov"],
        cutoff_angle_rad=first_lens["cutoff_angle_rad"],
    )
    return {
        "schema_version": 1,
        "source_freeze": {
            "commit": source["commit"],
            "tree": source["tree"],
            "tracked_count": source["tracked_count"],
        },
        "source_files": {
            "source_contract": str(source_contract_path),
            "source_contract_sha256": sha256_file(source_contract_path),
            "expanded_urdf": str(expanded_urdf_path),
            "expanded_urdf_sha256": sha256_file(expanded_urdf_path),
            "sdformat_camera_schema": str(sdformat_camera_schema_path),
            "sdformat_camera_schema_sha256": sha256_file(
                sdformat_camera_schema_path
            ),
        },
        "lens": {
            **first_lens,
            "effective_f_normalized": lens.effective_f,
            "continuous_image_center_uv": [lens.width / 2, lens.height / 2],
            "mapping": "r=c1*f*tan(theta/c2+c3)",
            "scale_to_hfov_rule": (
                "f=1/(c1*tan((horizontal_fov/2)/c2+c3))"
            ),
            "pixel_rule": (
                "u=W*(0.5+x_norm/2); "
                "v=H*(0.5-aspect*y_norm/2)"
            ),
        },
        "coordinate_convention": {
            "vectors": "column",
            "transform_chain": (
                "p_base=T_base_lidar*p_lidar; "
                "p_camera=inverse(T_base_camera)*p_base"
            ),
            "ros_sensor": "+X forward, +Y left, +Z up",
            "render_ray": "+X right, +Y up, +Z optical axis",
            "sensor_to_render": (
                "[x_render,y_render,z_render]=[-y_sensor,z_sensor,x_sensor]"
            ),
            "image": "+u right, +v down",
            "rpy": "fixed-axis Rz(yaw)*Ry(pitch)*Rx(roll), radians",
        },
        "metrics_preregistration": {
            "grayscale": "0.299R+0.587G+0.114B via cv2 RGB2GRAY",
            "gaussian_kernel": [5, 5],
            "gaussian_sigma": 1.2,
            "canny_low": 60,
            "canny_high": 120,
            "canny_l2gradient": True,
            "sobel_kernel": 3,
            "lidar_edge_jump": "max(0.5m, 0.05*nearer_range)",
            "nan_boundary_is_edge": False,
            "edge_support_px": [3.0, 5.0],
            "minimum_projected_depth_edges": 100,
            "low_evidence_policy": (
                "nominal 或 control 的投影深度边缘少于100时不进入胜负和bootstrap"
            ),
            "primary_score": (
                "control median nearest-image-edge distance - "
                "nominal median nearest-image-edge distance"
            ),
            "bootstrap_seed": 20260723,
            "bootstrap_replicates": 10000,
            "bootstrap_cluster": "bag x target_header_stamp",
            "minimum_eligible_clusters_for_pass": 18,
            "minimum_nominal_evidence_clusters_per_pair": 18,
        },
        "negative_controls": {
            "composition": (
                "T_control=Delta_T_camera*T_camera_lidar; "
                "每次只改变一个camera-frame自由度"
            ),
            "translation_m": {
                axis: [-0.5, 0.5] for axis in ("x", "y", "z")
            },
            "rotation_deg": {
                axis: [-10.0, 10.0] for axis in ("roll", "pitch", "yaw")
            },
            "cycle_mismatch": (
                "front->back->left->left2->right->right2->front"
            ),
            "axis_sign_error": (
                "错误地采用render_up=-sensor_z；其他参数不变"
            ),
            "no_search_or_optimization": True,
        },
        "nominal_transforms": transforms,
        "diagnostic_emx": {
            "T_base_forward_lidar": matrix_to_list(t_base_forward),
            "update_rate_hz": forward["update_rate_hz"],
            "lidar": forward["lidar"],
            "excluded_from_pairwise_pass_criterion": True,
        },
    }


def lens_from_contract(contract: dict[str, Any]) -> CustomTanLens:
    lens = contract["lens"]
    return CustomTanLens(
        width=int(lens["width"]),
        height=int(lens["height"]),
        horizontal_fov_rad=float(lens["horizontal_fov_rad"]),
        c1=float(lens["c1"]),
        c2=float(lens["c2"]),
        c3=float(lens["c3"]),
        configured_f=float(lens["configured_f"]),
        scale_to_hfov=bool(lens["scale_to_hfov"]),
        cutoff_angle_rad=float(lens["cutoff_angle_rad"]),
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML 根节点不是字典：{path}")
    return value


def decode_image(msg: Any) -> np.ndarray:
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"不支持的图像编码：{msg.encoding}")
    if int(msg.step) < int(msg.width) * 3:
        raise ValueError("图像 step 小于 width*3")
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = np.ndarray(
        (int(msg.height), int(msg.step)),
        dtype=np.uint8,
        buffer=raw,
    )
    image = rows[:, : int(msg.width) * 3].reshape(
        int(msg.height), int(msg.width), 3
    )
    if msg.encoding == "bgr8":
        image = image[:, :, ::-1]
    return image.copy()


def decode_pointcloud2(msg: Any) -> tuple[np.ndarray, dict[str, Any]]:
    fields = {field.name: field for field in msg.fields}
    for field in ("x", "y", "z"):
        if field not in fields:
            raise ValueError(f"PointCloud2 缺少字段：{field}")
        if int(fields[field].datatype) != 7:
            raise ValueError(f"PointCloud2 {field} 不是 FLOAT32")
    endian = ">" if bool(msg.is_bigendian) else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4"],
            "offsets": [
                int(fields["x"].offset),
                int(fields["y"].offset),
                int(fields["z"].offset),
            ],
            "itemsize": int(msg.point_step),
        }
    )
    raw = bytes(msg.data)
    required = int(msg.row_step) * int(msg.height)
    if len(raw) < required:
        raise ValueError("PointCloud2 data 短于 row_step*height")
    array = np.ndarray(
        shape=(int(msg.height), int(msg.width)),
        dtype=dtype,
        buffer=raw,
        strides=(int(msg.row_step), int(msg.point_step)),
    )
    points = np.stack(
        (
            array["x"].astype(np.float64),
            array["y"].astype(np.float64),
            array["z"].astype(np.float64),
        ),
        axis=-1,
    )
    return points, {
        "width": int(msg.width),
        "height": int(msg.height),
        "point_step": int(msg.point_step),
        "row_step": int(msg.row_step),
        "is_bigendian": bool(msg.is_bigendian),
        "fields": {
            name: {
                "offset": int(field.offset),
                "datatype": int(field.datatype),
                "count": int(field.count),
            }
            for name, field in fields.items()
        },
    }


def organized_geometry_edges(points: np.ndarray) -> np.ndarray:
    finite = np.isfinite(points).all(axis=2)
    ranges = np.linalg.norm(points, axis=2)
    edge = np.zeros(finite.shape, dtype=bool)

    right = np.roll(points, -1, axis=1)
    right_finite = np.roll(finite, -1, axis=1)
    right_range = np.roll(ranges, -1, axis=1)
    right_jump = np.linalg.norm(points - right, axis=2)
    right_threshold = np.maximum(0.5, 0.05 * np.minimum(ranges, right_range))
    right_edge = finite & right_finite & (right_jump > right_threshold)
    edge |= right_edge
    edge |= np.roll(right_edge, 1, axis=1)

    upper = points[:-1, :, :]
    lower = points[1:, :, :]
    pair_finite = finite[:-1, :] & finite[1:, :]
    jump = np.linalg.norm(upper - lower, axis=2)
    threshold = np.maximum(
        0.5, 0.05 * np.minimum(ranges[:-1, :], ranges[1:, :])
    )
    vertical_edge = pair_finite & (jump > threshold)
    edge[:-1, :] |= vertical_edge
    edge[1:, :] |= vertical_edge
    return edge


def image_edge_products(
    image_rgb: np.ndarray, lens: CustomTanLens
) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(
        blurred, threshold1=60, threshold2=120, L2gradient=True
    ) > 0
    valid_mask = lens.valid_pixel_mask()
    edges &= valid_mask
    inverse = np.full(edges.shape, 255, dtype=np.uint8)
    inverse[edges] = 0
    distance = cv2.distanceTransform(inverse, cv2.DIST_L2, 5)
    sobel_x = cv2.Sobel(
        blurred, cv2.CV_32F, 1, 0, ksize=3
    )
    sobel_y = cv2.Sobel(
        blurred, cv2.CV_32F, 0, 1, ksize=3
    )
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel[~valid_mask] = np.nan
    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "distance": distance,
        "sobel": sobel,
        "valid_mask": valid_mask,
    }


def zbuffer_indices(
    u: np.ndarray,
    v: np.ndarray,
    ranges: np.ndarray,
    valid: np.ndarray,
    image_width: int,
) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return indices
    pixel_u = np.floor(u[indices]).astype(np.int64)
    pixel_v = np.floor(v[indices]).astype(np.int64)
    linear = pixel_v * int(image_width) + pixel_u
    order = np.argsort(ranges[indices], kind="stable")
    sorted_indices = indices[order]
    sorted_linear = linear[order]
    _, first = np.unique(sorted_linear, return_index=True)
    return sorted_indices[np.sort(first)]


def delta_transform(
    dof: str, value: float, *, degrees: bool = False
) -> np.ndarray:
    delta = np.eye(4, dtype=float)
    if dof in ("x", "y", "z"):
        delta[("x", "y", "z").index(dof), 3] = value
        return delta
    angle = math.radians(value) if degrees else value
    rpy = {
        "roll": (angle, 0.0, 0.0),
        "pitch": (0.0, angle, 0.0),
        "yaw": (0.0, 0.0, angle),
    }
    if dof not in rpy:
        raise ValueError(f"未知扰动自由度：{dof}")
    delta[:3, :3] = rpy_to_matrix(*rpy[dof])
    return delta


if __name__ == "__main__":
    raise SystemExit(_module_cli())
