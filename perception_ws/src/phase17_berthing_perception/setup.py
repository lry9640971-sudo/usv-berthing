from setuptools import find_packages, setup


package_name = "phase17_berthing_perception"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", [
            "config/static_extrinsics.yaml",
            "config/full_dock_collision_points.csv",
            "config/berthing_perception.yaml",
            "config/hull_collision_footprint_base_link.yaml",
            "config/visual_self_geometry.yaml",
            "config/local_environment.yaml",
        ]),
        (f"share/{package_name}/launch", ["launch/berthing_perception.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "berthing_perception_node = phase17_berthing_perception.berthing_perception_node:main",
            "local_environment_node = phase17_berthing_perception.local_environment_node:main",
        ],
    },
)
