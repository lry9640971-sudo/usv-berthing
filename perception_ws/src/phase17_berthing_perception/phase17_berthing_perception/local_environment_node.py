"""五路 PointCloud2 普通障碍物与局部不可通行栅格节点。"""

from __future__ import annotations

from collections import defaultdict, deque
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import threading
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from phase16z_s3_runtime.s3_ownship_geometry import load_ownship_geometry

from .berthing_core import LIDAR_TOPICS, load_extrinsics
from .environment_core import (
    build_environment_grid,
    extract_cloud_evidence,
    inflate_occupied,
    load_footprint_polygons,
    self_mask_grid,
)
from .dock_localizer import load_template, rotate
from .grid_core import GridConfig, connected_components
from .input_sync import select_fresh_inputs
from .navigation_history import (
    TimestampedPoseBuffer, pose_from_odometry, planar_output_transform, snapshot_poses,
    world_xy_in_planar_frame,
)


GRID_TOPIC = "/team/perception/local_occupancy"
CENTERS_TOPIC = "/team/perception/obstacle_centers_local"
OBSTACLES_TOPIC = "/team/perception/ordinary_obstacles"
STATUS_TOPIC = "/team/perception/local_environment_status"
DOCK_TOPIC = "/team/perception/berth_candidates_world"


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class LocalEnvironmentNode(Node):
    def __init__(self) -> None:
        super().__init__("phase17_local_environment")
        share = Path(get_package_share_directory("phase17_berthing_perception"))
        self.declare_parameter("extrinsics_path", str(share / "config" / "static_extrinsics.yaml"))
        self.declare_parameter(
            "footprint_path", str(share / "config" / "hull_collision_footprint_base_link.yaml")
        )
        self.declare_parameter("self_geometry_path", str(share / "config" / "visual_self_geometry.yaml"))
        self.declare_parameter(
            "dock_template_path", str(share / "config" / "full_dock_collision_points.csv")
        )
        self.declare_parameter("receipt_path", "")
        self.declare_parameter("resolution_m", 0.20)
        self.declare_parameter("width", 400)
        self.declare_parameter("height", 400)
        self.declare_parameter("origin_x_m", -40.0)
        self.declare_parameter("origin_y_m", -40.0)
        self.declare_parameter("minimum_range_m", 0.5)
        self.declare_parameter("maximum_range_m", 40.0)
        self.declare_parameter("mean_water_elevation_m", 0.)
        self.declare_parameter("sync_tolerance_ms", 100.0)
        self.declare_parameter("minimum_lidar_inputs", 3)
        self.declare_parameter("maximum_lidar_age_ms", 350.0)
        self.declare_parameter("inflation_radius_m", 0.4)
        self.declare_parameter("minimum_occupied_cell_hits", 2)
        self.declare_parameter("angular_ray_bin_deg", 0.5)
        self.declare_parameter("output_stale_s", 0.5)
        self.declare_parameter("dock_fusion_stale_s", 0.5)
        self.declare_parameter("base_frame", "wamv/base_link_planar")

        self.extrinsics = load_extrinsics(str(self.get_parameter("extrinsics_path").value))
        self.footprint = load_footprint_polygons(str(self.get_parameter("footprint_path").value))
        self.self_geometry = load_ownship_geometry(str(self.get_parameter("self_geometry_path").value))
        self.dock_template, _dock_groups = load_template(
            Path(str(self.get_parameter("dock_template_path").value))
        )
        receipt_text = str(self.get_parameter("receipt_path").value).strip()
        self.receipt_path = Path(receipt_text) if receipt_text else None
        self.cfg = GridConfig(
            resolution=float(self.get_parameter("resolution_m").value),
            width=int(self.get_parameter("width").value),
            height=int(self.get_parameter("height").value),
            origin_xy=(
                float(self.get_parameter("origin_x_m").value),
                float(self.get_parameter("origin_y_m").value),
            ),
            min_range=float(self.get_parameter("minimum_range_m").value),
            max_range=float(self.get_parameter("maximum_range_m").value),
            variant="V0_OCCUPIED_FIRST",
        )
        self.sync_tolerance_ns = int(float(self.get_parameter("sync_tolerance_ms").value) * 1e6)
        self.minimum_lidar_inputs = int(self.get_parameter("minimum_lidar_inputs").value)
        self.maximum_lidar_age_ns = int(
            float(self.get_parameter("maximum_lidar_age_ms").value) * 1e6
        )
        if not 1 <= self.minimum_lidar_inputs <= len(LIDAR_TOPICS) or self.maximum_lidar_age_ns <= 0:
            raise RuntimeError("LIDAR_FRESHNESS_PARAMETER_INVALID")
        self.inflation_radius_m = float(self.get_parameter("inflation_radius_m").value)
        self.minimum_occupied_cell_hits = int(
            self.get_parameter("minimum_occupied_cell_hits").value
        )
        self.angular_ray_bin_deg = float(self.get_parameter("angular_ray_bin_deg").value)
        self.mean_water_elevation_m = float(self.get_parameter("mean_water_elevation_m").value)
        self.output_stale_s = float(self.get_parameter("output_stale_s").value)
        self.dock_fusion_stale_ns = int(
            float(self.get_parameter("dock_fusion_stale_s").value) * 1e9
        )
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.self_mask = self_mask_grid(self.cfg, self.footprint)

        # 在线点云不排队补发历史帧，两个最新测量已足够覆盖同步邻域。
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        output_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.grid_publisher = self.create_publisher(OccupancyGrid, GRID_TOPIC, output_qos)
        self.centers_publisher = self.create_publisher(PoseArray, CENTERS_TOPIC, output_qos)
        self.obstacles_publisher = self.create_publisher(String, OBSTACLES_TOPIC, output_qos)
        self.status_publisher = self.create_publisher(String, STATUS_TOPIC, output_qos)
        self.input_subscriptions = [
            self.create_subscription(
                PointCloud2,
                topic,
                lambda message, topic_name=topic: self._cloud_callback(message, topic_name),
                qos,
            )
            for topic in LIDAR_TOPICS
        ]
        self.dock_subscription = self.create_subscription(
            String, DOCK_TOPIC, self._dock_callback, output_qos
        )
        self.navigation_subscription = self.create_subscription(
            Odometry, "/team/localization/odometry", self._navigation_callback, output_qos)
        self.status_timer = self.create_timer(0.5, self._publish_status)

        self.lock = threading.RLock()
        self.latest_clouds: dict[str, PointCloud2] = {}
        self.latest_dock: dict | None = None
        self.last_enqueued_stamp_ns = 0
        self.navigation_buffer = TimestampedPoseBuffer()
        self.last_output_stamp_ns = 0
        self.last_output_wall = 0.0
        self.last_result: dict | None = None
        self.last_reason = "WAITING_FOR_FIVE_LIDARS"
        self.counts: dict[str, int] = defaultdict(int)
        self.queue: Queue[dict | None] = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, name="environment-latest-frame", daemon=True)
        self.worker.start()
        self.started_wall = time.time()
        self.get_logger().info("局部环境节点已启动：等待五路有组织 PointCloud2")

    def _dock_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            pose = np.asarray(payload["dock_pose_xyyaw_m_rad"], dtype=float)
            valid = (payload.get("status") == "READY" and payload.get("frame_id") == "team_enu"
                     and pose.shape == (3,) and np.isfinite(pose).all())
            item = {"stamp_ns": int(payload["stamp_ns"]), "pose": pose, "valid": bool(valid)}
        except Exception:
            self.counts["DOCK_MESSAGE_INVALID"] += 1
            return
        with self.lock:
            self.latest_dock = item
            self.counts["DOCK_MESSAGE_VALID"] += int(item["valid"])

    def _cloud_callback(self, message: PointCloud2, topic: str) -> None:
        with self.lock:
            self.latest_clouds[topic] = message
            self.counts[topic] += 1
            self._enqueue_snapshot_locked()

    def _enqueue_snapshot_locked(self):
        selection = select_fresh_inputs(
            {
                name: stamp_ns(cloud.header.stamp)
                for name, cloud in self.latest_clouds.items()
            },
            minimum_inputs=self.minimum_lidar_inputs,
            maximum_age_ns=self.maximum_lidar_age_ns,
        )
        if selection is None:
            self.counts["INSUFFICIENT_FRESH_LIDARS"] += 1
            return
        selected_names, anchor, span_ns = selection
        if anchor <= self.last_enqueued_stamp_ns:
            return
        full_sync = (
            len(selected_names) == len(LIDAR_TOPICS)
            and span_ns <= self.sync_tolerance_ns
        )
        if not full_sync:
            self.counts["PARTIAL_OR_ASYNC_LIDAR_FRAME"] += 1
        clouds = tuple((name, self.latest_clouds[name]) for name in LIDAR_TOPICS if name in selected_names)
        try:
            poses = snapshot_poses(self.navigation_buffer, clouds, anchor)
        except ValueError:
            self.last_reason = "WAIT_NAVIGATION_INTERPOLATION_ENDPOINT"
            self.counts[self.last_reason] += 1
            return
        snapshot = {
            "poses": poses,
            "anchor": anchor,
            "span_ms": span_ns / 1e6,
            "clouds": tuple(
                (name, self.latest_clouds[name])
                for name in LIDAR_TOPICS if name in selected_names
            ),
            "input_mode": "FULL_SYNC" if full_sync else "PARTIAL_FRESH",
        }
        try:
            self.queue.put_nowait(snapshot)
        except Full:
            try:
                self.queue.get_nowait()
            except Empty:
                pass
            self.queue.put_nowait(snapshot)
            self.counts["LATEST_FRAME_REPLACED"] += 1
        self.last_enqueued_stamp_ns = anchor

    @staticmethod
    def _read_xyz(message: PointCloud2) -> np.ndarray:
        values = point_cloud2.read_points_numpy(
            message, field_names=["x", "y", "z"], skip_nans=False
        )
        if getattr(values.dtype, "names", None):
            return np.column_stack((values["x"], values["y"], values["z"])).astype(np.float64)
        return np.asarray(values, dtype=np.float64).reshape(-1, 3)

    def _navigation_callback(self, message):
        try:
            pose = pose_from_odometry(message)
        except ValueError:
            with self.lock:
                self.counts["NAVIGATION_POSE_REJECTED"] += 1
            return
        with self.lock:
            self.navigation_buffer.add(pose)
            self.navigation_buffer.discard_before(pose.stamp_ns-2_000_000_000)
            self._enqueue_snapshot_locked()

    def _process_snapshot(self, snapshot: dict) -> None:
        started = time.monotonic()
        anchor_pose = snapshot["poses"][snapshot["anchor"]]
        occupied_parts = []
        finite_rays = []
        free_only_rays = []
        sensors = {}
        for topic, message in snapshot["clouds"]:
            points = self._read_xyz(message)
            source_pose = snapshot["poses"][stamp_ns(message.header.stamp)]
            occupied, sensor_finite, sensor_free, stats = extract_cloud_evidence(
                points,
                width=int(message.width),
                height=int(message.height),
                transform=self.extrinsics[topic],
                footprint_polygons=self.footprint,
                self_geometry=self.self_geometry,
                cfg=self.cfg,
                angular_bin_deg=self.angular_ray_bin_deg,
                navigation_z_m=source_pose.z_m,
                gravity_rotation=Rotation.from_quat(source_pose.quaternion_xyzw).as_matrix(),
                output_transform=planar_output_transform(source_pose, anchor_pose),
                mean_water_elevation_m=self.mean_water_elevation_m,
            )
            occupied_parts.append(occupied)
            finite_rays.extend(sensor_finite)
            free_only_rays.extend(sensor_free)
            sensors[topic] = stats
        extraction_finished = time.monotonic()
        grid, compact, stats = build_environment_grid(
            occupied_parts,
            finite_rays,
            free_only_rays,
            cfg=self.cfg,
            self_mask=self.self_mask,
            inflation_radius_m=self.inflation_radius_m,
            minimum_occupied_cell_hits=self.minimum_occupied_cell_hits,
        )
        grid_finished = time.monotonic()

        dock_fused = False
        with self.lock:
            dock = self.latest_dock
        if (
            dock is not None
            and dock["valid"]
            and abs(int(snapshot["anchor"]) - int(dock["stamp_ns"])) <= self.dock_fusion_stale_ns
        ):
            x_m, y_m, yaw = [float(value) for value in dock["pose"]]
            # 世界系码头位姿保持原测量时间；先到 ENU，再按栅格锚点转入水平船体系。
            # 不能把较早的相对船体位姿直接套在转向后的另一帧中。
            dock_world = rotate(self.dock_template, yaw) + np.asarray([x_m, y_m])
            dock_points = world_xy_in_planar_frame(dock_world, anchor_pose)
            cells = np.floor(
                (dock_points - np.asarray(self.cfg.origin_xy)) / self.cfg.resolution
            ).astype(np.int64)
            valid_cells = (
                (cells[:, 0] >= 0) & (cells[:, 0] < self.cfg.width)
                & (cells[:, 1] >= 0) & (cells[:, 1] < self.cfg.height)
            )
            dock_grid = np.full_like(grid, -1)
            selected = cells[valid_cells]
            dock_grid[selected[:, 1], selected[:, 0]] = 100
            dock_grid = inflate_occupied(dock_grid, self.cfg, self.inflation_radius_m)
            dock_occupied = (dock_grid == 100) & ~self.self_mask
            grid[dock_occupied] = 100
            grid[(dock_grid == 50) & (grid == -1) & ~self.self_mask] = 50
            compact, _grid_only = connected_components(grid, self.cfg)
            dock_fused = True

        stamp = int(snapshot["anchor"])
        grid_message = OccupancyGrid()
        grid_message.header.stamp.sec = int(stamp // 1_000_000_000)
        grid_message.header.stamp.nanosec = int(stamp % 1_000_000_000)
        grid_message.header.frame_id = self.base_frame
        grid_message.info.resolution = float(self.cfg.resolution)
        grid_message.info.width = int(self.cfg.width)
        grid_message.info.height = int(self.cfg.height)
        grid_message.info.origin.position.x = float(self.cfg.origin_xy[0])
        grid_message.info.origin.position.y = float(self.cfg.origin_xy[1])
        grid_message.info.origin.orientation.w = 1.0
        grid_message.data = [int(value) for value in grid.reshape(-1)]
        self.grid_publisher.publish(grid_message)

        ordered = sorted(compact, key=lambda item: (item["centroid_xy_m"][0], item["centroid_xy_m"][1]))
        centers_message = PoseArray()
        centers_message.header = grid_message.header
        obstacles = []
        for index, item in enumerate(ordered, start=1):
            pose = Pose()
            pose.position.x = float(item["centroid_xy_m"][0])
            pose.position.y = float(item["centroid_xy_m"][1])
            pose.orientation.w = 1.0
            centers_message.poses.append(pose)
            obstacles.append(
                {
                    "obstacle_id": f"ordinary_{index}",
                    "center_xy_m": list(item["centroid_xy_m"]),
                    "bbox_min_xy_m": list(item["bbox_min_xy_m"]),
                    "bbox_max_xy_m": list(item["bbox_max_xy_m"]),
                    "area_m2": float(item["area_m2"]),
                    "cell_count": int(item["cell_count"]),
                    "valid": True,
                }
            )
        self.centers_publisher.publish(centers_message)
        obstacle_payload = {
            "schema": "team/ordinary_obstacles_v1",
            "stamp_ns": stamp,
            "frame_id": self.base_frame,
            "status": "READY",
            "input_mode": snapshot["input_mode"],
            "active_lidar_count": len(snapshot["clouds"]),
            "obstacles": obstacles,
        }
        self.obstacles_publisher.publish(
            String(data=json.dumps(obstacle_payload, ensure_ascii=False, separators=(",", ":")))
        )

        elapsed = time.monotonic() - started
        result = {
            "valid": True,
            "processing_s": elapsed,
            "phase_s": {"extraction": extraction_finished-started,
                        "grid": grid_finished-extraction_finished, "publication": elapsed-(grid_finished-started)},
            "measurement_age_sim_s": (self.get_clock().now().nanoseconds-stamp)/1e9,
            "sync_span_ms": float(snapshot["span_ms"]),
            "input_mode": snapshot["input_mode"],
            "active_lidar_count": len(snapshot["clouds"]),
            "ordinary_obstacle_count": len(obstacles),
            "dock_geometry_fused": dock_fused,
            "raw_occupied_cells": int(stats["raw_occupied_cells"]),
            "physical_occupied_cells": int(np.count_nonzero(grid == 100)),
            "inflated_occupied_cells": int(np.count_nonzero(grid >= 50)),
            "padding_cells": int(np.count_nonzero(grid == 50)),
            "free_cells": int(np.count_nonzero(grid == 0)),
            "unknown_cells": int(np.count_nonzero(grid == -1)),
            "sensors": sensors,
        }
        with self.lock:
            self.last_output_stamp_ns = stamp
            self.last_output_wall = time.monotonic()
            self.last_result = result
            self.last_reason = (
                "FIVE_LIDAR_GRID_VALID"
                if snapshot["input_mode"] == "FULL_SYNC"
                else "PARTIAL_FRESH_LIDAR_GRID_VALID"
            )
            self.counts["OUTPUT_FRAMES"] += 1

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                snapshot = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if snapshot is None:
                break
            try:
                self._process_snapshot(snapshot)
            except Exception as exc:
                with self.lock:
                    self.last_reason = f"PROCESSING_ERROR:{type(exc).__name__}"
                    self.counts[self.last_reason] += 1
                self.get_logger().error(f"局部环境帧处理失败: {exc}")

    def _publish_status(self) -> None:
        with self.lock:
            age = time.monotonic() - self.last_output_wall if self.last_output_wall else math.inf
            ready = bool(age <= self.output_stale_s and self.last_result)
            payload = {
                "schema": "team/local_environment_status_v1",
                "status": "READY" if ready else "DEGRADED",
                "reason": self.last_reason if ready else (
                    "OUTPUT_STALE" if self.last_output_wall else self.last_reason
                ),
                "last_output_stamp_ns": self.last_output_stamp_ns,
                "output_age_wall_s": age if math.isfinite(age) else None,
                "inputs_seen": dict(self.counts),
                "last_result": self.last_result,
            }
        self.status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except Full:
            pass
        self.worker.join(timeout=10.0)
        if self.receipt_path:
            with self.lock:
                receipt = {
                    "schema": "phase17/local_environment_receipt_v1",
                    "started_wall_unix_s": self.started_wall,
                    "finished_wall_unix_s": time.time(),
                    "counts": dict(self.counts),
                    "last_reason": self.last_reason,
                    "last_result": self.last_result,
                    "production_truth_topics_subscribed": [],
                    "control_topics_published": [],
                }
            self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            self.receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    rclpy.init()
    node = LocalEnvironmentNode()
    # 回调保持短小，后台线程处理点云；在线采样显示线程池调度占据主要预算。
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def stop_handler(*_) -> None:
        executor.shutdown(timeout_sec=2.0)

    signal.signal(signal.SIGTERM, stop_handler)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        try:
            executor.remove_node(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
