/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:39:10
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-07 15:38:45
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "yolo/yolo.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;

bool Yolo::readModel(Net& net, string& netPath, bool isCuda = false) {
    try {
        net = readNet(netPath);
    } catch (const std::exception&) {
        return false;
    }
    // cuda
    if (isCuda) {
        LOG_F(INFO, "use gpu");
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    // cpu
    else {
        LOG_F(INFO, "use cpu");
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    return true;
}
bool Yolo::Detect(Mat& SrcImg, Net& net, vector<Output>& output) {
    output.clear();
    Mat blob;
    int col = SrcImg.cols;
    int row = SrcImg.rows;
    // int maxLen = MAX(col, row);
    int maxLen      = max(col, row);
    Mat netInputImg = SrcImg.clone();
    if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
        Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
        SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
        netInputImg = resizeImg;
    }
    blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight),
                  cv::Scalar(0, 0, 0), true, false);
    //如果在其他设置没有问题的情况下但是结果偏差很大，可以尝试下用下面两句语句
    // blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth,
    // netHeight), cv::Scalar(104, 117, 123), true, false);
    // blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth,
    // netHeight), cv::Scalar(114, 114,114), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> netOutputImg;
    // vector<string> outputLayerName{"345","403", "461","output" };
    // net.forward(netOutputImg, outputLayerName[3]); //获取output的输出
    net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
    std::vector<int> classIds;        //结果id数组
    std::vector<float> confidences;   //结果每个id对应置信度数组
    std::vector<cv::Rect> boxes;      //每个id矩形框
    float ratio_h = (float)netInputImg.rows / netHeight;
    float ratio_w = (float)netInputImg.cols / netWidth;
    int net_width = className.size() + 5;   //输出的网络宽度是类别数+5
    float* pdata  = (float*)netOutputImg[0].data;
    for (int stride = 0; stride < strideSize; stride++) {   // stride
        int grid_x = (int)(netWidth / netStride[stride]);
        int grid_y = (int)(netHeight / netStride[stride]);
        for (int anchor = 0; anchor < 3; anchor++) {   // anchors
            // const float anchor_w = netAnchors[stride][anchor * 2];
            // const float anchor_h = netAnchors[stride][anchor * 2 + 1];
            for (int i = 0; i < grid_y; i++) {
                for (int j = 0; j < grid_x; j++) {
                    float box_score = pdata[4];
                    ;   //获取每一行的box框中含有某个物体的概率
                    if (box_score >= boxThreshold) {
                        cv::Mat scores(1, className.size(), CV_32FC1,
                                       pdata + 5);
                        Point classIdPoint;
                        double max_class_socre;
                        minMaxLoc(scores, 0, &max_class_socre, 0,
                                  &classIdPoint);
                        max_class_socre = (float)max_class_socre;
                        if (max_class_socre >= classThreshold) {
                            // rect [x,y,w,h]
                            float x  = pdata[0];   // x
                            float y  = pdata[1];   // y
                            float w  = pdata[2];   // w
                            float h  = pdata[3];   // h
                            int left = (x - 0.5 * w) * ratio_w;
                            int top  = (y - 0.5 * h) * ratio_h;
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back(max_class_socre * box_score);
                            boxes.push_back(Rect(left, top, int(w * ratio_w),
                                                 int(h * ratio_h)));
                        }
                    }
                    pdata += net_width;   //下一行
                }
            }
        }
    }

    //执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
    vector<int> nms_result;
    NMSBoxes(boxes, confidences, nmsScoreThreshold, nmsThreshold, nms_result);
    for (size_t i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Output result;
        result.id         = classIds[idx];
        result.confidence = confidences[idx];
        result.box        = boxes[idx];
        output.push_back(result);
    }
    if (output.size())
        return true;
    else
        return false;
}

bool isInside(Rect rect1, Rect rect2) {
    return (rect1 == (rect1 & rect2));
}

cv::Mat Yolo::drawPred(Mat& img, vector<Output> result, vector<Scalar> color) {
    for (size_t i = 0; i < result.size(); i++) {
        rectangle(img, result[i].box, color[result[i].id], 2, 8);
    }
    return img;
}
