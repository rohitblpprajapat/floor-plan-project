cmake_minimum_required(VERSION 3.16)
project(metric_360_vslam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Optional Ceres and GTSAM
find_package(Ceres QUIET)
find_package(GTSAM QUIET)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

if (Ceres_FOUND)
    include_directories(${CERES_INCLUDE_DIRS})
    message(STATUS "Found Ceres: ${CERES_VERSION}")
endif()

if (GTSAM_FOUND)
    include_directories(${GTSAM_INCLUDE_DIR})
    message(STATUS "Found GTSAM: ${GTSAM_VERSION}")
endif()

# Core library
add_library(metric_vslam_core INTERFACE)
target_include_directories(metric_vslam_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

message(STATUS "Metric 360 VSLAM CMake Configuration Complete.")
