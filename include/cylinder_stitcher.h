/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:53:47
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-04-26 17:07:06
 * @Description: 
 * 
 * Copyright (c) 2024 by pandaman, All Rights Reserved. 
 */
#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <thrust/extrema.h>
#include "math_utils.h"

#include "helper_cuda.h"
#include "loguru.hpp"

struct CylinderImageGPU {
    uchar3 *image;
    uchar *mask;
    int *uv;
    int height, width;

    CylinderImageGPU() {}

    CylinderImageGPU(uchar3 *image_, uchar *mask_, int height_, int width_) {
        image = image_;
        mask = mask_;
        height = height_;
        width = width_;

        cudaMalloc((void **) &uv, sizeof(int) * 2);
        cudaMemset(uv, 0, sizeof(int) * 2);
    }

    CylinderImageGPU(uchar3 *image_, uchar *mask_, int *uv_, int height_, int width_) {
        image = image_;
        mask = mask_;
        uv = uv_;
        height = height_;
        width = width_;
    }

    CylinderImageGPU(int height_, int width_) {
        height = height_;
        width = width_;
        cudaMalloc((void **) &image, sizeof(uchar3) * height * width);
        cudaMalloc((void **) &mask, sizeof(uchar) * height * width);
        cudaMalloc((void **) &uv, sizeof(int) * 2);
        cudaMemset(uv, 0, sizeof(int) * 2);
    }

    bool toCPU(cv::Mat &image_, cv::Mat &mask_) {
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        uchar3 *data_rgb;
        uchar *data_mask;

        printf("toCPU() height: %d, width: %d\n", height, width);

        checkCudaErrors(cudaHostAlloc((void **) &data_rgb, sizeof(uchar3) * height * width, cudaHostAllocDefault));
        checkCudaErrors(cudaHostAlloc((void **) &data_mask, sizeof(uchar) * height * width, cudaHostAllocDefault));

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(data_rgb, image, sizeof(uchar3) * height * width, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(data_mask, mask, sizeof(uchar) * height * width, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        image_ = cv::Mat(height, width, CV_8UC3, data_rgb);
        mask_ = cv::Mat(height, width, CV_8UC1, data_mask);

        image_ = image_.clone();
        mask_ = mask_.clone();
        cudaFreeHost(data_rgb);
        cudaFreeHost(data_mask);
        return true;
    }
};


struct SeamImageGPU {
    short3 *image;
    short *mask;
    int height, width;

    SeamImageGPU() {}


    SeamImageGPU(short3 *image_, short *mask_, int height_, int width_) {
        image = image_;
        mask = mask_;
        height = height_;
        width = width_;
    }

    void clear() {
        cudaFree(image);
        cudaFree(mask);
    }
};
