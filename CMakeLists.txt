cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

SET(CMAKE_BUILD_TYPE Debug)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

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

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()


find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(SophusTempl REQUIRED)
find_package(PCL REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/
${PROJECT_SOURCE_DIR}/Thirdparty/alglib/include/
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${SophusTempl_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_definitions(-D_DISABLE_LOOP_CLOSURE_)
#option(_DISABLE_LOOP_CLOSURE_ "Disable the loop closure." ON)

#add_library(${PROJECT_NAME} SHARED
#src/System.cc
#src/Tracking.cc
#src/LocalMapping.cc
#src/LoopClosing.cc
#src/ORBextractor.cc
#src/ORBmatcher.cc
#src/FrameDrawer.cc
#src/Converter.cc
#src/MapPoint.cc
#src/KeyFrame.cc
#src/Map.cc
#src/MapDrawer.cc
#src/Optimizer.cc
#src/PnPsolver.cc
#src/Frame.cc
#src/KeyFrameDatabase.cc
#src/Sim3Solver.cc
#src/Initializer.cc
#src/Viewer.cc
#src/semantic_lab.cpp
#src/semantic_classifier.cpp
#src/semantic_g2o_staff.cpp
#)
#
#target_link_libraries(${PROJECT_NAME}
#${OpenCV_LIBS}
#${EIGEN3_LIBS}
#${Pangolin_LIBRARIES}
#${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
#${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
#)

# Build examples

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

# add_executable(rgbd_tum
# Examples/RGB-D/rgbd_tum.cc)
# target_link_libraries(rgbd_tum ${PROJECT_NAME})

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

# add_executable(stereo_kitti
# Examples/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti ${PROJECT_NAME})

# add_executable(stereo_euroc
# Examples/Stereo/stereo_euroc.cc)
# target_link_libraries(stereo_euroc ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

# add_executable(mono_tum
# Examples/Monocular/mono_tum.cc)
# target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        src/semantic_lab.cpp
        src/semantic_classifier.cpp
        src/semantic_g2o_staff.cpp
        src/object.cpp
        src/ellipsoid.cpp)

target_link_libraries(mono_kitti
   ${OpenCV_LIBS}
   ${EIGEN3_LIBS}
   ${Pangolin_LIBRARIES}
   ${PCL_LIBRARIES}
   ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
   ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
   ${PROJECT_SOURCE_DIR}/Thirdparty/alglib/lib/libALGLIB.so
)

# add_executable(mono_euroc
# Examples/Monocular/mono_euroc.cc)
# target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(SemanticTest 
src/semantic_lab.cpp
src/semantic_classifier.cpp
test/test_semantic.cpp)
target_link_libraries(SemanticTest ${OpenCV_LIBS})

add_executable(SemanticMemoryUsageTest 
src/semantic_lab.cpp
src/semantic_classifier.cpp
test/test_semantic_memory_usage.cpp)
target_link_libraries(SemanticMemoryUsageTest ${OpenCV_LIBS})

add_executable(EllipsoidTest
src/ellipsoid.cpp
test/test_ellipsoid.cpp)

add_executable(ObjSLAMTest
src/ellipsoid.cpp
src/object.cpp
src/object_detection.cpp
src/quadric_g2o_staff.cpp)

add_executable(ObjectDATest
        test/test_object.cpp)
target_link_libraries(ObjectDATest
        ${PROJECT_SOURCE_DIR}/Thirdparty/alglib/lib/libALGLIB.so
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES})