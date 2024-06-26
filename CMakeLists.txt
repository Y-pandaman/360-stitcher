cmake_minimum_required(VERSION 3.16)

project(pano_code)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wno-unknown-pragmas -Wall -Ofast -fPIC -pthread ${CMAKE_CXX_FLAGS}")

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

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${eCAL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

include_directories(${PROJECT_SOURCE_DIR}/plugin/ ${PROJECT_SOURCE_DIR}/include/)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)
add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

cuda_add_executable(
        pano_test
        apps/main.cpp
)

CUDA_ADD_LIBRARY(pano_main SHARED
        src/core/airview_stitcher.cpp
        src/core/airview_stitcher_kernel.cu
        src/stage/blend_kernel.cu
        src/stage/undistorter.cpp
        src/stage/gst_receiver.cpp
        src/util/loguru.cpp
        src/yolo/yolo.cpp
        src/yolo/yolo_detect.cpp
        src/stage/ecal_image_sender.cpp
        src/core/pano_main.cpp
        src/stage/camera_sender.cpp
        src/util/config.cpp
        src/stage/render.cu
        src/yolo/calibrator.cpp
        src/yolo/model.cpp
        src/yolo/postprocess.cpp
        src/yolo/preprocess.cu
        src/yolo/yolov5trt_det.cpp
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
        myplugins
)
