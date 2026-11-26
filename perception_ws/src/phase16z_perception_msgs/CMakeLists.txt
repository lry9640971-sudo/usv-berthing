cmake_minimum_required(VERSION 3.8)
project(phase16z_perception_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DynamicDetection.idl"
  "msg/DynamicDetectionArray.idl"
  "msg/DynamicTrack.idl"
  "msg/DynamicTrackArray.idl"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
