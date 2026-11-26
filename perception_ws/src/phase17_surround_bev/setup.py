from glob import glob
from setuptools import find_packages, setup


package_name = "phase17_surround_bev"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
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
            "surround_bev_node = phase17_surround_bev.surround_bev_node:main",
            "calibration_contract_check = phase17_surround_bev.calibration_contract_check:main",
            "auto_extrinsic_calibration = phase17_surround_bev.auto_extrinsic_calibration:main",
        ],
    },
)
