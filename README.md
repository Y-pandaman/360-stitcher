- 编译

  ```
  mkdir build
  cd build
  cmake ..
  make -j
  ```

- 使用

  ```
  ./pano_test ../assets/ 0
  ```

- 文件结构

  ```
  ├── 使用说明.md
  ├── apps
  │   └── main.cpp
  ├── assets
  │   ├── camera_video_0429   // 录制的视频
  │   ├── camera_video1  // 录制的视频
  │   ├── car.png  // 叠加的车模型
  │   ├── pano_assist.png
  │   ├── weights   // 权重
  │   └── yamls  // 内外参
  ├── CMakeLists.txt
  ├── include
  │   ├── common
  │   ├── core
  │   ├── stage
  │   └── util
  ├── output   // 输出的拼接视频
  │   └── video.avi
  ├── proto
  │   └── image.proto
  └── src
      ├── core
      ├── stage
      └── util
  ```

- 可选优化

  - yolo模型可以换成tensorrt的，能够加速一部分
  - 部分for循环可以使用parallel_for_并行加速
  - 当前的gst接收必须等6个相机全部有数据才会开始拼接，若一个相机坏了，收不到数据，会一直卡在那里，
  - gst接收程序部分缓冲区会一直保存最后一帧，即使没数据，也会一直拼最后一帧
  - 目前叠加车模型或者倒车辅助线都是在gpu里做的

- `pano_main.cpp`中的开关：

  ```
  #define OUTPUT_STITCHING_RESULT_VIDEO  // 显示+保存拼接图像
  // #define USE_GST_INPUT   // 使用gst实时流
  // #define RESEND_ORIGINAL_IMAGE  // 转发特定几个视野的画面
  #define USE_VIDEO_INPUT  // 使用录制的视频
  #define USE_720P  // 使用720P画面（原相机是1080P的，内参也是1080P的，视频流是转成720P再发出来的）
  ```

  