# Third-party notices

The rebuilt planner and controller, berthing perception, ROS 2 interfaces,
and evaluation records were contributed by [lry9640971-sudo](https://github.com/lry9640971-sudo).
The original competition effort was collaborative. This repository carries no
repository-wide license. Existing copyright notices and component licenses
remain applicable.

| Material | Source and license |
| --- | --- |
| Perception and interface packages under `perception_ws/src/` | The package manifests declare Apache-2.0. The [Apache-2.0 text](LICENSES/Apache-2.0.txt) applies to material covered by these declarations. |
| ROS 2 template tests under `control_ws/src/competition_user/test/` | `test_copyright.py`, `test_flake8.py`, and `test_pep257.py` retain their original Open Source Robotics Foundation, Inc. copyright notices and Apache-2.0 headers. |
| [Competition USVSim](https://gitee.com/little-dolphin-intelligence/2026-challenge-cup) and [Virtual RobotX](https://github.com/osrf/vrx) | External simulation platform. The recorded USVSim revision is `75babf08680736cb932b8e078fdeae1531a10134` (v1.0.5). Its VRX source subtree carries Apache-2.0. Install the platform and its resources separately under their own terms. |
| ROS 2, Gazebo, NumPy, SciPy, Shapely, OpenCV, PyYAML, and pytest | Installed dependencies. Refer to each upstream distribution for its license and notices. |

The legacy team controller and the tests that depend on it are omitted. The
run scores and small evaluation inputs are from the original project simulations.

## External runtime assets

Use copies you are authorized to obtain and use. Place each file at its
original path relative to the repository root before running the build
and regressions. The original delivery contains those files. The USVSim
checkout alone does not supply these derived configurations or learned
models. The BEV lookup table is needed for the optional camera module
when enabled. The files below are excluded from this repository.

| Path | Used by |
| --- | --- |
| `control_ws/src/competition_user/config/vessel_dynamics.yaml` | Vessel mass, inertia and actuator parameters for the controller and its tests. |
| `perception_ws/src/phase16z_s3_runtime/config/dynamic_semantic_background_v2.json` | Offline vessel classifier used by the dynamic tracking launch. |
| `perception_ws/src/phase16z_s3_runtime/config/hull_collision_footprint_base_link.yaml` | Ownship collision footprint used by LiDAR candidate filtering. |
| `perception_ws/src/phase16z_s3_runtime/config/model_sensor_mounts.yaml` | Sensor mounts used by GPS/IMU navigation. |
| `perception_ws/src/phase16z_s3_runtime/config/visual_self_geometry.yaml` | Ownship visual geometry used by LiDAR candidate filtering. |
| `perception_ws/src/phase17_berthing_perception/config/full_dock_collision_points.csv` | Dock template used by registration and local mapping. |
| `perception_ws/src/phase17_berthing_perception/config/hull_collision_footprint_base_link.yaml` | Ownship collision footprint used by perception, control and tests. |
| `perception_ws/src/phase17_berthing_perception/config/visual_self_geometry.yaml` | Ownship visual geometry used by environment and dock perception. |
| `perception_ws/src/phase17_surround_bev/config/masked_fixed_geometry_bev_lut.npz` | Lookup table for the optional camera surround view. |
