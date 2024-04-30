#pragma once

#include "math_utils.h"    // 在上
#include "helper_cuda.h"   // 在下
#include <opencv2/opencv.hpp>

/**
 * 表示一个圆柱体图像的GPU数据结构。
 * 包含图像数据、掩码数据、UV坐标、图像高度和宽度等成员。
 */
struct CylinderImageGPU {
    uchar3* image;   // 图像数据，uchar3表示RGB三通道颜色
    uchar* mask;   // 图像掩码数据，用于标识图像中的有效或无效像素
    int* uv;             // UV坐标数据，用于纹理映射
    int height, width;   // 图像的高度和宽度

    /**
     * 默认构造函数。
     */
    CylinderImageGPU() { }

    /**
     * 构造函数，初始化图像、掩码、高度、宽度，并在GPU上分配UV坐标内存。
     *
     * @param image_ 输入图像数据指针。
     * @param mask_ 输入掩码数据指针。
     * @param height_ 图像高度。
     * @param width_ 图像宽度。
     */
    CylinderImageGPU(uchar3* image_, uchar* mask_, int height_, int width_) {
        image  = image_;
        mask   = mask_;
        height = height_;
        width  = width_;

        cudaMalloc((void**)&uv, sizeof(int) * 2);
        cudaMemset(uv, 0, sizeof(int) * 2);
    }

    /**
     * 构造函数，初始化图像、掩码、UV坐标、高度、宽度。
     *
     * @param image_ 输入图像数据指针。
     * @param mask_ 输入掩码数据指针。
     * @param uv_ 输入UV坐标数据指针。
     * @param height_ 图像高度。
     * @param width_ 图像宽度。
     */
    CylinderImageGPU(uchar3* image_, uchar* mask_, int* uv_, int height_,
                     int width_) {
        image  = image_;
        mask   = mask_;
        uv     = uv_;
        height = height_;
        width  = width_;
    }

    /**
     * 构造函数，根据给定的高度和宽度在GPU上分配图像、掩码和UV坐标内存。
     *
     * @param height_ 图像高度。
     * @param width_ 图像宽度。
     */
    CylinderImageGPU(int height_, int width_) {
        height = height_;
        width  = width_;
        cudaMalloc((void**)&image, sizeof(uchar3) * height * width);
        cudaMalloc((void**)&mask, sizeof(uchar) * height * width);
        cudaMalloc((void**)&uv, sizeof(int) * 2);
        cudaMemset(uv, 0, sizeof(int) * 2);
    }

    /**
     * 将GPU上的图像和掩码数据拷贝到CPU的OpenCV Mat对象中。
     *
     * @param image_ 输出图像的OpenCV Mat对象引用。
     * @param mask_ 输出掩码的OpenCV Mat对象引用。
     * @return 总是返回true。
     */
    bool toCPU(cv::Mat& image_, cv::Mat& mask_) {
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        uchar3* data_rgb;
        uchar* data_mask;

        checkCudaErrors(cudaHostAlloc((void**)&data_rgb,
                                      sizeof(uchar3) * height * width,
                                      cudaHostAllocDefault));
        checkCudaErrors(cudaHostAlloc((void**)&data_mask,
                                      sizeof(uchar) * height * width,
                                      cudaHostAllocDefault));

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(data_rgb, image,
                                   sizeof(uchar3) * height * width,
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(data_mask, mask,
                                   sizeof(uchar) * height * width,
                                   cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        // 将数据赋值给传入的Mat对象，并克隆以确保数据拥有独立内存
        image_ = cv::Mat(height, width, CV_8UC3, data_rgb);
        mask_  = cv::Mat(height, width, CV_8UC1, data_mask);

        image_ = image_.clone();
        mask_  = mask_.clone();
        cudaFreeHost(data_rgb);
        cudaFreeHost(data_mask);
        return true;
    }
};

/**
 * 结构体 SeamImageGPU 用于表示在GPU上存储的缝合图像及其对应的掩码。
 */
struct SeamImageGPU {
    short3*
        image;   // 指向GPU内存中存储的图像数据的指针，每个像素由一个short3表示
    short* mask;   // 指向GPU内存中存储的图像掩码的指针，每个像素由一个short表示
    int height, width;   // 图像的高度和宽度

    // 默认构造函数
    SeamImageGPU() { }

    /**
     * 构造函数，初始化SeamImageGPU结构体。
     *
     * @param image_ 指向GPU上图像数据的指针。
     * @param mask_ 指向GPU上图像掩码的指针。
     * @param height_ 图像的高度。
     * @param width_ 图像的宽度。
     */
    SeamImageGPU(short3* image_, short* mask_, int height_, int width_) {
        image  = image_;
        mask   = mask_;
        height = height_;
        width  = width_;
    }

    /**
     * 释放SeamImageGPU结构体中指向的GPU内存。
     */
    void clear() {
        cudaFree(image);   // 释放图像数据的GPU内存
        cudaFree(mask);    // 释放图像掩码的GPU内存
    }
};
