#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

void CudaPreprocessInit(int max_image_size);
void CudaPreprocessDestroy();
void CudaPreprocess(uint8_t* src, int src_width, int src_height,
                     float* dst, int dst_width, int dst_height,
                     cudaStream_t stream);
void CudaBatchPreprocess(std::vector<cv::Mat>& img_batch,
                           float* dst, int dst_width, int dst_height,
                           cudaStream_t stream);

