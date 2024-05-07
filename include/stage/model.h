#pragma once

#include <NvInfer.h>
#include <string>

nvinfer1::ICudaEngine* BuildDetEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                        nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                        float& gd, float& gw, std::string& wts_name);

nvinfer1::ICudaEngine* BuildDetP6Engine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder,
                                           nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt,
                                           float& gd, float& gw, std::string& wts_name);

nvinfer1::ICudaEngine* BuildClsEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, float& gd, float& gw, std::string& wts_name);

nvinfer1::ICudaEngine* BuildSegEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, float& gd, float& gw, std::string& wts_name);
