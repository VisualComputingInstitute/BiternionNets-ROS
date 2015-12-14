cmake_minimum_required(VERSION 2.8.3)
project(biternion)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    geometry_msgs
    image_transport
    message_filters
    roscpp
    std_msgs
    message_generation
    rwth_upper_body_detector
    rwth_pedestrian_tracking
)

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  HeadOrientations.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
)

###########
## Build ##
###########

# We know that for STRANDS and SPENCER it's g++4.6 anyways.
# And we can't use CMake's new feature for C++11 yet.
add_definitions(-std=c++0x)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare a cpp executable
add_executable(count_ubd src/count_ubd.cpp)
add_executable(dump_ubd src/dump_ubd.cpp)
add_executable(dump_tracks src/dump_tracks.cpp src/mkpath.c src/subtractbg.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
add_dependencies(count_ubd ${PROJECT_NAME}_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
add_dependencies(dump_ubd ${PROJECT_NAME}_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
add_dependencies(dump_tracks ${PROJECT_NAME}_generate_messages_cpp ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(count_ubd ${catkin_LIBRARIES})
target_link_libraries(dump_ubd ${catkin_LIBRARIES})
target_link_libraries(dump_tracks ${catkin_LIBRARIES})

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS dump_ubd dump_tracks count_ubd
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# TODO: What's up with python scripts?

install(DIRECTORY models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
