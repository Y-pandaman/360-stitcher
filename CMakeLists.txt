cmake_minimum_required(VERSION 3.16)
#project(360code LANGUAGES CXX)
project(pano_code)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -fPIC -pthread")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(include
        )

# opencv
find_package(OpenCV REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

# eigen3
include_directories(
        "/usr/include/eigen3/"
)

# cuda
find_package(CUDA REQUIRED)


#cuda_add_executable(pano
#        include/airview_stitcher.h
#        include/airview_stitcher_utils.h
#        include/blend.h
#        include/cylinder_stitcher.h
#        include/math_utils.h
#        include/helper_cuda.h
#        include/helper_string.h
#        src/airview_stitcher.cpp
#        src/airview_stitcher_kernel.cu
#        src/blend_kernel.cu
#        src/cylinder_stitcher.cpp
#        src/Undistorter.cpp
#        src/GstReceiver.cpp
#        src/main.cpp
#        src/loguru.cpp
#        src/yolo.cpp
#        src/yolo_detect.cpp
#        src/Intersector.cpp
#        src/EcalImageSender.cpp
#        )

cuda_add_executable(
    pano_test
        src/main.cpp
        )

cuda_add_executable(
    detect_test
        src/yolo.cpp
        src/loguru.cpp
        src/yolo_detect.cpp
        src/detect_test_main.cpp)

cuda_add_executable(
    image_stitch_test
        src/airview_stitcher.cpp
        src/airview_stitcher_kernel.cu
        src/blend_kernel.cu
        src/cylinder_stitcher.cpp
        src/Undistorter.cpp
        src/GstReceiver.cpp
        src/loguru.cpp
        src/yolo.cpp
        src/yolo_detect.cpp
        src/Intersector.cpp
        src/image_stitch_test_main.cpp)

CUDA_ADD_LIBRARY(pano_main SHARED
        include/airview_stitcher.h
        include/airview_stitcher_utils.h
        include/blend.h
        include/cylinder_stitcher.h
        include/math_utils.h
        include/helper_cuda.h
        include/helper_string.h
        src/airview_stitcher.cpp
        src/airview_stitcher_kernel.cu
        src/blend_kernel.cu
        src/cylinder_stitcher.cpp
        src/Undistorter.cpp
        src/GstReceiver.cpp
#        src/main.cpp
        src/loguru.cpp
        src/yolo.cpp
        src/yolo_detect.cpp
        src/Intersector.cpp
        src/EcalImageSender.cpp include/PanoMain.h src/PanoMain.cpp src/CameraSender.cpp src/CameraSender.h)


# ecal & protobuf
find_package(eCAL REQUIRED)
include_directories(${eCAL_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
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
        eCAL::core
)

target_link_libraries(
    image_stitch_test
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        eCAL::core
)

target_link_libraries(
    detect_test
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        eCAL::core
)