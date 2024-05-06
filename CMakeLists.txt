cmake_minimum_required(VERSION 3.16)

project(pano_code)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wno-unknown-pragmas -Wall -fPIC -pthread")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(eCAL REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Freetype REQUIRED)
find_package(OpenMP)

if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${eCAL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

cuda_add_executable(
        pano_test
        apps/main.cpp
)

CUDA_ADD_LIBRARY(pano_main SHARED
        src/airview_stitcher.cpp
        src/airview_stitcher_kernel.cu
        src/blend_kernel.cu
        src/Undistorter.cpp
        src/GstReceiver.cpp
        src/loguru.cpp
        src/yolo.cpp
        src/yolo_detect.cpp
        src/EcalImageSender.cpp
        src/PanoMain.cpp
        src/CameraSender.cpp
        src/Config.cpp
        src/project_to_cylinder.cu
        src/render.cu
)

set(protobuf_files
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/image.proto
)
protobuf_target_cpp(pano_test proto ${protobuf_files})
protobuf_target_cpp(pano_main proto ${protobuf_files})

target_link_libraries(
        pano_test
        pano_main
)

target_link_libraries(
        pano_main
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        ${EIGEN3_LIBRARIES}
        ${FREETYPE_LIBRARIES}
        eCAL::core
)
