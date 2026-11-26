from glob import glob
from setuptools import setup

package_name = "phase16z_s3_runtime"

setup(
    name=package_name,
    version="0.3.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "s3_lidar_candidate_node = phase16z_s3_runtime.s3_lidar_candidate_node:main",
            "s3_lidar_pointcloud_candidate_node = phase16z_s3_runtime.s3_lidar_pointcloud_candidate_node:main",
            "s3_localization_dynamic_tracks = phase16z_s3_runtime.s3_localization_dynamic_tracks:main",
            "s3_navigation_node = phase16z_s3_runtime.s3_navigation_node:main",
            "s3_gps_velocity_bridge = phase16z_s3_runtime.s3_gps_velocity_bridge:main",
            "test_ros_inputs = phase16z_s3_runtime.test_ros_inputs:main",
            "dynamic_tracks_consumer = phase16z_s3_runtime.dynamic_tracks_consumer:main",
            "fixed_sensor_tf_node = phase16z_s3_runtime.fixed_sensor_tf_node:main",
        ],
    },
)
