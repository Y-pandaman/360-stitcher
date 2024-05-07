/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:51:36
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-07 15:40:04
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#pragma once

#include "yolo/yolo.h"
#include "util/loguru.hpp"
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

private:
    Yolo yolo;
    cv::dnn::Net net;
    std::vector<cv::Scalar> color;
    std::vector<Output> result;
    bool use_CUDA = true;
};
}   // namespace yolo_detect
