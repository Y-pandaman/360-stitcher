#include "yolo_detect.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
int main() {
    cv::VideoCapture capture;
    std::filesystem::path video_path =
        "/home/xcmg/stitcher/360-stitcher/video/pano_video.mp4";
    if (!capture.open(video_path)) {
        printf("cannot open %s\n", video_path.c_str());
        return 0;
    }

    yolo_detect::YoloDetect yolo_detect(
        "/home/xcmg/stitcher/360-stitcher/parameters/weights/best.onnx", true);
    while (true) {
        cv::Mat image;
        bool flag = capture.read(image);
        if (!flag)
            break;
        yolo_detect.detect(image);
        cv::imshow("test", image);
        cv::waitKey(0);
    }
}