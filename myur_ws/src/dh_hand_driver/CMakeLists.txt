cmake_minimum_required(VERSION 2.8.3)
project(dh_hand_driver)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  serial
  std_msgs
  message_generation
  actionlib_msgs
  actionlib
  genmsg
)

add_action_files(
  DIRECTORY action
  FILES ActuateHand.action
)

add_service_files(
  DIRECTORY srv 
  FILES hand_state.srv
)

add_message_files(
  FILES
  Pos_state.msg
)

generate_messages(
  DEPENDENCIES 
  actionlib_msgs 
  std_msgs
)

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    message_runtime
    roscpp
    rospy
    serial
    actionlib_msgs
    std_msgs
)

# add_library(LIBS

# )

include_directories(
  ${catkin_INCLUDE_DIRS}
  include
  include/dh_hand_driver
)

#add_library(${PROJECT_NAME} SHARED 
#include/dh_hand_driver/hand_controller.cpp
#include/dh_hand_driver/DH_datastream.cpp 
#include/dh_hand_driver/hand_driver.cpp 
#)


add_executable(hand_controller   
include/dh_hand_driver/hand_controller.cpp
include/dh_hand_driver/DH_datastream.cpp 
include/dh_hand_driver/hand_driver.cpp 
src/node.cpp)

add_dependencies(hand_controller
  dh_hand_driver_generate_messages_cpp
  ${dh_hand_driver_EXPORTED_TARGETS})

target_link_libraries(hand_controller 
${catkin_LIBRARIES})

add_executable(hand_controller_client   
src/test_client.cpp)
target_link_libraries(hand_controller_client  ${catkin_LIBRARIES})

