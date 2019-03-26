cmake_minimum_required(VERSION 2.8.3)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -lfftw3 -lm")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
FIND_PACKAGE(Eigen REQUIRED)
FIND_PACKAGE(catkin REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

include_directories(
    include
    include/feat_extra
    include/implements
    ${Eigen_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_library( IMP_SRC
        src/config.cpp
        src/matlab_func.cpp
        src/feat_extra/fhog.cpp
        src/feat_extra/cnf.cpp

        src/implements/tracker.cpp
        src/implements/train.cpp
        src/implements/scale_filter.cpp
        src/implements/sample_space.cpp
		)	# TODO shared?
# add_dependencies(IMP_SRC ${OpenCV_LIBRARIES}
#                             m)


add_executable(func_test func_test.cpp)       
target_link_libraries(func_test
                    IMP_SRC
                    ${OpenCV_LIBRARIES}
                    fftw3
                    m
                    )

add_executable(eco main.cpp)       
target_link_libraries(eco 
                    IMP_SRC
                    ${OpenCV_LIBRARIES}
                    fftw3
                    m
                    )

add_executable(vot100_test vot100_test.cpp)       
target_link_libraries(vot100_test 
                    IMP_SRC
                    ${OpenCV_LIBRARIES}
                    fftw3
                    m
                    )

add_executable(video_test test_720p.cpp)       
target_link_libraries(video_test 
                    IMP_SRC
                    ${OpenCV_LIBRARIES}
                    fftw3
                    m
                    )
