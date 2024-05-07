#include "yolo/yolo_detect.h"

namespace yolo_detect {
YoloDetect::YoloDetect(std::string model_path, bool use_CUDA) {
    if (yolo.readModel(net, model_path, use_CUDA)) {
        LOG_F(INFO, "read net ok!");
    } else {
        LOG_F(INFO, "read net failed!");
        return;
    }
}

YoloDetect::~YoloDetect() { }

cv::Mat YoloDetect::detect(cv::Mat img) {
    cv::Mat result_img;
    //生成随机颜色
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(b, g, r));
    }

    if (yolo.Detect(img, net, result)) {
        result_img = yolo.drawPred(img, result, color);
        return result_img;
    } else {
        //        LOG_F(INFO, "Detect Failed!");
        return img;
    }
}

struct struct_yolo_result YoloDetect::detect_bbox(cv::Mat img) {
    cv::Mat result_img;
    struct struct_yolo_result result_;
    //生成随机颜色
    srand(time(0));
    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(b, g, r));
    }

    if (yolo.Detect(img, net, result)) {
        //        result_img = yolo.drawPred(img, result, color);
        result_.img    = result_img;
        result_.result = result;
        return result_;
        // return result_img;
    } else {
        //        LOG_F(INFO, "Detect Failed!");
        // return img;
        result_.img = img;
        return result_;
    }
}

}   // namespace yolo_detect
