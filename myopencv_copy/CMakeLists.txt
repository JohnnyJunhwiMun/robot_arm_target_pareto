cmake_minimum_required(VERSION 3.0.2)
project(myopencv_copy)

find_package(catkin REQUIRED
  COMPONENTS
    opencv_services
    interactive_markers
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_ros_perception
    rviz_visual_tools
    moveit_visual_tools
    pluginlib
    geometric_shapes
    pcl_ros
    pcl_conversions
    rosbag
    tf2_ros
    tf2_eigen
    tf2_geometry_msgs
    actionlib_msgs
    control_msgs
    roscpp
    rospy
    message_generation

)

# Add these lines for Python and pybind11
find_package(PythonLibs 3.6 REQUIRED)  # Adjust version as needed
find_package(pybind11 REQUIRED)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  IntList.msg
  floatList.msg
)


# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


generate_messages(
  DEPENDENCIES actionlib_msgs std_msgs trajectory_msgs control_msgs
)


catkin_package(
  LIBRARIES
  INCLUDE_DIRS
  CATKIN_DEPENDS
    moveit_core
    moveit_visual_tools
    moveit_ros_planning_interface
    interactive_markers
    tf2_geometry_msgs
    actionlib_msgs
    control_msgs
    opencv_services
  DEPENDS
    PYTHON  # Add this line
)

include_directories(SYSTEM 
  ${catkin_INCLUDE_DIRS} 
  ${Boost_INCLUDE_DIR} 
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}  # Add this line
  ${pybind11_INCLUDE_DIRS}  # Add this line
)

# Your existing executables

add_executable(normal_improved_pareto2 src/normal_improved_pareto2.cpp)
target_link_libraries(normal_improved_pareto2 ${catkin_LIBRARIES} ${Boost_LIBRARIES})


# Copy Python script to the build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/jacobian_func.py
               ${CMAKE_CURRENT_BINARY_DIR}/jacobian_func.py COPYONLY)

# # Copy Python script to the build directory
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/ThreadPool.py
# ${CMAKE_CURRENT_BINARY_DIR}/ThreadPool.py COPYONLY)

# # Optionally, you can also install the Python script
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/jacobian_func.py
#         DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# install(TARGETS pick_and_place_opencv DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
# install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})