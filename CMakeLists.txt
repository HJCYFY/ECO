cmake_minimum_required(VERSION 2.8.3)
project(eco)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

find_package(catkin REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
)

# file(GLOB  SOURCES ${CMAKE_SOURCE_DIR}/src/feature_extraction/*.cpp)

add_library(source  
            src/feature_extraction/hog.cpp)	# TODO shared?


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)


add_executable(test_ensure src/test_ensure.cpp)       
target_link_libraries(test_ensure 
                    source
                    ${OpenCV_LIBRARIES})