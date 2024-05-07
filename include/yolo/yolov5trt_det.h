/*
 * @Author: 姚潘涛
 * @Date: 2024-05-07 14:31:22
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-07 15:40:18
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "util/loguru.hpp"
#include "yolo/cuda_utils.h"
#include "yolo/logging.h"
#include "yolo/model.h"
#include "yolo/postprocess.h"
#include "yolo/preprocess.h"
#include "yolo/utils.h"
#include "yolo/yolo.h"
#include "yolo/yolo_detect.h"
#include <chrono>
#include <cmath>
#include <iostream>

#define USE_IMG_DIR false
#define USE_CAMERA true
#define SERIALIZE_ENGINE false

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize =
    kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

class Yolov5TrtDet {
public:
    Yolov5TrtDet(std::string engine);
    ~Yolov5TrtDet();
    void PrepareBuffers(ICudaEngine* engine, float** gpu_input_buffer,
                        float** gpu_output_buffer, float** cpu_output_buffer);
    void Infer(IExecutionContext& context, cudaStream_t& stream,
               void** gpu_buffers, float* output, int batchsize);
    void SerializeEngine(unsigned int max_batchsize, bool is_p6, float gd,
                         float gw, std::string wts_name,
                         std::string engine_name);
    void DeserializeEngine(std::string& engine_name, IRuntime** runtime,
                           ICudaEngine** engine, IExecutionContext** context);
    void Detect(std::vector<cv::Mat> imgs, std::vector<cv::Mat>& out_imgs);
    void DetectImgDir(std::string dir);
    void DetectCamera(cv::Mat img, cv::Mat& result_img);
    struct struct_yolo_result detect_bbox(cv::Mat img);

private:
    std::string wts_name_    = "";
    std::string engine_name_ = "";
    std::string net_         = "";
    bool is_p6_              = false;
    float gd_ = 0.0f, gw_ = 0.0f;
    std::string img_dir_;

    cudaStream_t stream_;

    float* gpu_buffers_[2];
    float* cpu_output_buffer_   = nullptr;
    IRuntime* runtime_          = nullptr;
    ICudaEngine* engine_        = nullptr;
    IExecutionContext* context_ = nullptr;
};