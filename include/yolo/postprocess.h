#pragma once

#include "types.h"
#include "yolo_detect.h"
#include <opencv2/opencv.hpp>

cv::Rect getRect(cv::Mat& img, float bbox[4]);

void Nms(std::vector<Detection>& res, float* output, float conf_thresh,
         float nms_thresh = 0.5);

void BatcNms(std::vector<std::vector<Detection>>& batch_res, float* output,
             int batch_size, int output_size, float conf_thresh,
             float nms_thresh = 0.5);

void DrawBbox(std::vector<cv::Mat>& img_batch,
              std::vector<std::vector<Detection>>& res_batch);

void DrawBboxWithResult(std::vector<cv::Mat>& img_batch,
                        std::vector<std::vector<Detection>>& res_batch,
                        struct_yolo_result& result);

std::vector<cv::Mat> ProcessMask(const float* proto, int proto_size,
                                 std::vector<Detection>& dets);

void DrawMaskBbox(cv::Mat& img, std::vector<Detection>& dets,
                  std::vector<cv::Mat>& masks,
                  std::unordered_map<int, std::string>& labels_map);
