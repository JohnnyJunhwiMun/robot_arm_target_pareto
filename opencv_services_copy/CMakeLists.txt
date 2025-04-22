cmake_minimum_required(VERSION 3.0.2)
project(opencv_services_copy)

## Compile as C++17, supported in ROS Melodic and newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  tf2_geometry_msgs  # Add this line
  geometry_msgs
  message_generation
  pcl_conversions  # Add pcl_conversions
  pcl_ros          # Add pcl_ros
  visualization_msgs # Ensure visualization_msgs is included
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
)

## Find OpenCV, Eigen, and PCL
find_package(OpenCV 4 REQUIRED)  # Ensure you have OpenCV 4
find_package(Eigen3 REQUIRED)    # Find Eigen (header-only library)
find_package(PCL REQUIRED)       # PCL is needed for point cloud processing

## Generate services in the 'srv' folder
add_service_files(
  FILES
  all_normal.srv
)

## Generate added messages and services with dependencies
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs cv_bridge image_transport tf2_ros tf2_geometry_msgs geometry_msgs message_runtime 
  pcl_ros pcl_conversions visualization_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
)

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

## Declare executables

add_executable(axes_plane_all src/axes_plane_all.cpp)



## Add dependencies for message generation

add_dependencies(axes_plane_all opencv_services_generate_messages_cpp)

## Link libraries to executables
target_link_libraries(axes_plane_all ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${pcl_ros_LIBRARIES} ${pcl_conversions_LIBRARIES} )
