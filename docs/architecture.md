# ROS 2 interfaces and frames

The competition platform supplies five `PointCloud2` streams (front, back, left, right, and forward), GPS fixes and velocity, IMU data, and simulation time. The runs used 10 Hz LiDAR. The forward sensor publishes a 360 × 96 scan, and the other four publish 360 × 48 scans.

| Topic | Type | Frame or use |
| --- | --- | --- |
| `/team/localization/odometry` | `nav_msgs/msg/Odometry` | Pose in `team_enu`. Twist is expressed in the body frame identified by `child_frame_id`. |
| `/team/perception/dynamic_detections_base` | `phase16z_perception_msgs/msg/DynamicDetectionArray` | LiDAR candidates in the vessel frame. The tracker uses odometry at the detection timestamp. |
| `/team/perception/dynamic_tracks` | `phase16z_perception_msgs/msg/DynamicTrackArray` | Tracked positions and absolute velocities in `team_enu`. |
| `/team/perception/local_occupancy` | `nav_msgs/msg/OccupancyGrid` | Local grid with free, occupied, and unknown cells. The timestamp and frame identify the projection into the controller's map. |
| `/world/obstacles/dock` | `geometry_msgs/msg/PoseStamped` | Coarse dock prior supplied by the competition platform. |
| `/team/perception/berth_candidates_world` | `std_msgs/msg/String` carrying JSON | Refined berth poses and occupancy states. The controller uses valid, recent observations to select and maintain its target. |
| `/wamv/thrusters/{left,right}/thrust` | `std_msgs/msg/Float64` | Left and right force commands under the simulator actuator contract. |
| `/team/planning/status`, `/team/planning/completion` | `std_msgs/msg/String` carrying JSON | Mission state, timing, and completion information. |

The [controller entry point](../control_ws/src/competition_user/competition_user/closed_loop/node.py) connects the mission state machine and planner to the thrust executor. The controller consumes observed state and the published dock prior. The [independent monitor](../scripts/official_monitor.py) reads Gazebo model poses and `/vrx/contacts` for scoring. It also records the controller's completion report so the evaluator can align measurements with the report timestamp.

The detection and track types are generated from the checked-in IDL. If `ros2 interface show` on Jazzy fails to render these definitions, read the [message sources](../perception_ws/src/phase16z_perception_msgs/msg) directly. The generated Python types are available through `phase16z_perception_msgs.msg`.
