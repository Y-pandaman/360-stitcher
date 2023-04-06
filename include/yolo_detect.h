#pragma once

#include "yolo.h"
#include "loguru.hpp"
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
struct struct_yolo_result {
    cv::Mat img;
    std::vector<Output> result;
};

namespace yolo_detect {
class YoloDetect {
public:
    YoloDetect(std::string model_path, bool use_CUDA);
    ~YoloDetect();
    cv::Mat detect(cv::Mat img);
    struct struct_yolo_result detect_bbox(cv::Mat img);
//    std::string get_class_name(int id){return yolo.cla}
private:
    Yolo yolo;
    cv::dnn::Net net;
    std::vector<cv::Scalar> color;
    std::vector<Output> result;
    bool use_CUDA = true;
};
} // namespace yolo_detect

