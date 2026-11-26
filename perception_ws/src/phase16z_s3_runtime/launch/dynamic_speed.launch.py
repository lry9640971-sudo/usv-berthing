"""启动冻结的五路点云动态船检测与绝对速度链。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("phase16z_s3_runtime"))
    config = share / "config"
    use_sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("candidate_receipt_path", default_value=""),
            DeclareLaunchArgument("localization_receipt_path", default_value=""),
            DeclareLaunchArgument("navigation_receipt_path", default_value=""),
            DeclareLaunchArgument("gps_velocity_gz_topic", default_value="/world/berthing_user/model/wamv/link/wamv/gps_wamv_link/sensor/navsat/navsat"),
            DeclareLaunchArgument("gps_velocity_ros_topic", default_value="/wamv/sensors/gps/gps/velocity"),
            DeclareLaunchArgument("enable_gps_velocity", default_value="true"),
            DeclareLaunchArgument("maximum_detection_range_m", default_value="80.0"),
            DeclareLaunchArgument("publish_fixed_sensor_tf", default_value="false"),
            Node(
                package="phase16z_s3_runtime",
                executable="fixed_sensor_tf_node",
                name="phase16z_fixed_sensor_tf",
                output="screen",
                condition=IfCondition(LaunchConfiguration("publish_fixed_sensor_tf")),
                parameters=[{"use_sim_time": use_sim_time}],
            ),
            Node(
                package="phase16z_s3_runtime",
                executable="s3_lidar_pointcloud_candidate_node",
                name="phase16z_dynamic_candidate",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "receipt_path": LaunchConfiguration("candidate_receipt_path"),
                        "footprint_path": str(config / "hull_collision_footprint_base_link.yaml"),
                        "tf_proof_path": str(config / "fixed_sensor_tf_proof.json"),
                        "voxel_size_m": 0.35,
                        "cluster_radius_m": 0.55,
                        "organized_dual_path": True,
                        "minimum_vertical_consistent_neighbors": 2,
                        "cluster_center_mode": "pca_bbox",
                        "cluster_center_radial_offset_m": 0.2,
                        "sync_tolerance_ms": 100.0,
                        "maximum_detection_range_m": LaunchConfiguration("maximum_detection_range_m"),
                    }
                ],
            ),
            Node(
                package="phase16z_s3_runtime",
                executable="s3_navigation_node",
                name="phase16z_navigation",
                output="screen",
                parameters=[str(config / "sensor_navigation.yaml"), {
                    "use_sim_time": use_sim_time,
                    "receipt_path": LaunchConfiguration("navigation_receipt_path"),
                    "tf_proof_path": str(config / "fixed_sensor_tf_proof.json"),
                    "frame_contract_path": str(config / "team_enu_frame_contract.json"),
                    "gps_velocity_topic": LaunchConfiguration("gps_velocity_ros_topic"),
                }],
            ),
            Node(
                package="phase16z_s3_runtime",
                executable="s3_gps_velocity_bridge",
                output="screen",
                condition=IfCondition(LaunchConfiguration("enable_gps_velocity")),
                parameters=[{"use_sim_time": use_sim_time,
                             "gz_topic": LaunchConfiguration("gps_velocity_gz_topic")}],
            ),
            Node(
                package="phase16z_s3_runtime",
                executable="s3_localization_dynamic_tracks",
                name="phase16z_dynamic_tracking",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "receipt_path": LaunchConfiguration("localization_receipt_path"),
                        "semantic_filter_enabled": True,
                        "semantic_model_path": str(config / "dynamic_semantic_background_v2.json"),
                        # 公共轨迹只允许经过稳定语义确认的对象出生。低阈值会把
                        # 码头边缘在自船运动下产生的短时关联误当作动态船。
                        "semantic_primary_minimum_hit_count": 9,
                        "semantic_birth_probability": 0.9,
                        "semantic_association_position_gate_m": 2.0,
                        "semantic_association_velocity_gate_mps": 1.0,
                        "semantic_publication_coast_s": 0.7,
                        "semantic_velocity_previous_weight": 0.25,
                        # 点云同步可能有 1--2 秒间隔；保留跨帧证据，避免
                        # 真实动态船每次都被当成新轨迹而无法出生。
                        "tracker_maximum_track_age_s": 4.0,
                        "semantic_minimum_birth_evidence_hits": 2,
                        "semantic_birth_evidence_max_gap_s": 2.5,
                        "semantic_maximum_background_frames": 8,
                        # 稀疏近水面回波交给多帧语义层判别，不能在 ENU 入口永久删除。
                        "minimum_candidate_enu_z_m": -0.25,
                    }
                ],
            ),
        ]
    )
