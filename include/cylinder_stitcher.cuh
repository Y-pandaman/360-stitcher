#pragma once

//#define OUTPUT_CYL_IMAGE

#include "math_utils.h"    // 在上

#include "helper_cuda.h"   // 在下
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <string>
#include <thrust/extrema.h>
/**/
struct PinholeCameraGPU_stilib {
    float fx, fy, cx, cy;
    float d0, d1, d2, d3;
    float* R = nullptr;
    float* T = nullptr;
    float* C = nullptr;

    PinholeCameraGPU_stilib() { }

    PinholeCameraGPU_stilib(float _fx, float _fy, float _cx, float _cy,
                            float _d0, float _d1, float _d2, float _d3,
                            float* _R, float* _T, float* _C) {
        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;
        d0 = _d0;
        d1 = _d1;
        d2 = _d2;
        d3 = _d3;
        R  = _R;
        T  = _T;
        C  = _C;
    }

    // 将三维空间中的向量 v 绕一个固定的轴进行旋转
    inline __device__ float3 rotateVector(float3 v) {
        return make_float3(R[0] * v.x + R[1] * v.y + R[2] * v.z,
                           R[3] * v.x + R[4] * v.y + R[5] * v.z,
                           R[6] * v.x + R[7] * v.y + R[8] * v.z);
    }
    // 获取中心坐标
    inline __device__ float3 getCenter() {
        return make_float3(C[0], C[1], C[2]);
    }

    // 将三维世界坐标系中的点 x 转换（投影）到二维图像平面上的像素坐标
    inline __device__ float2 projWorldToPixel(float3 x) {
        // 将世界坐标 x 减去相机的光心坐标 C，得到相对于相机光心的坐标
        x.x -= C[0];
        x.y -= C[1];
        x.z -= C[2];
        float3 cam = this->rotateVector(x);   // 将相对坐标 x 旋转到相机坐标系
        // 如果旋转后的z坐标小于0，表示该点位于相机的背面，无法投影到图像平面上
        if (cam.z < 0)
            return make_float2(-1, -1);
        // 使用透视投影公式将其从相机坐标系转换为像素坐标
        // 计算点在图像宽度和高度方向上的像素坐标
        return make_float2(cam.x * fx / cam.z + cx, cam.y * fy / cam.z + cy);
    }

    // 将三维世界坐标系中的点 x 转换（投影）到鱼眼相机的图像平面上的像素坐标
    inline __device__ float2 projWorldToPixelFishEye(float3 x) {
        // 将世界坐标 x 减去相机的光心坐标 C，得到相对于相机光心的坐标
        x.x -= C[0];
        x.y -= C[1];
        x.z -= C[2];
        float3 cam = this->rotateVector(x);   // 将相对坐标 x 旋转到相机坐标系

        // 如果旋转后的z坐标小于0，表示该点不在相机的前方，无法投影到图像平面上
        if (cam.z < 0)
            return make_float2(-1, -1);

        // 通过齐次坐标进行透视除法，将三维点转换为二维点 (xx, yy)
        float xx = cam.x / cam.z, yy = cam.y / cam.z;
        // 计算二维点到光心的欧几里得距离 r
        float r = sqrtf(xx * xx + yy * yy);
        // 计算从光心到二维点的射线与 z 轴正方向之间的夹角 theta
        float theta = atan(r);
        // 计算 theta 的二到八次幂，用于后续的畸变模型计算
        float theta2 = theta * theta, theta4 = theta2 * theta2,
              theta6 = theta4 * theta2, theta8 = theta4 * theta4;
        // 根据畸变系数 d0 到 d3 和 theta 的幂次，计算畸变校正后的角 theta_d
        float theta_d =
            theta * (1 + d0 * theta2 + d1 * theta4 + d2 * theta6 + d3 * theta8);
        // 计算缩放因子 scale，如果 r 为0（即点在光心上），则缩放因子为1；
        // 否则，缩放因子为畸变校正后的角除以原始角 r
        float scale = (r == 0) ? 1.0 : theta_d / r;
        // 应用缩放因子 scale 到二维点 (xx, yy)，然后乘以相机的内参（焦距 fx, fy
        // 和光心坐标 cx, cy），得到最终的像素坐标
        return make_float2(fx * xx * scale + cx, fy * yy * scale + cy);
    }
};

struct ViewGPU_stilib {
    uchar3* image = nullptr;
    uchar* mask   = nullptr;
    int height, width;
    PinholeCameraGPU_stilib camera;

    uchar3* data_rgb_mem_buffer = nullptr;
    uchar* data_mask_mem_buffer = nullptr;

    ViewGPU_stilib() { }

    ViewGPU_stilib(uchar3* image_, uchar* mask_, int height_, int width_,
                   PinholeCameraGPU_stilib camera_) {
        image  = image_;
        mask   = mask_;
        height = height_;
        width  = width_;
        camera = camera_;
    }

    /**
     * 将GPU上的图像和掩码数据拷贝到CPU的内存中。
     *
     * @param image_ 指向CV::Mat的引用，用于接收从GPU拷贝过来的图像数据。
     * @param mask_ 指向CV::Mat的引用，用于接收从GPU拷贝过来的掩码数据。
     * @return 总是返回true。
     */
    bool toCPU(cv::Mat& image_, cv::Mat& mask_) {
        // 如果相应的内存缓冲区尚未分配，则进行分配
        if (data_rgb_mem_buffer == nullptr)
            cudaHostAlloc((void**)&data_rgb_mem_buffer,
                          sizeof(uchar3) * height * width,
                          cudaHostAllocDefault);
        if (data_mask_mem_buffer == nullptr)
            cudaHostAlloc((void**)&data_mask_mem_buffer,
                          sizeof(uchar) * height * width, cudaHostAllocDefault);

        // 如果图像数据非空，则从GPU拷贝到CPU的内存缓冲区，并创建对应的CV::Mat对象
        if (image != nullptr) {
            cudaMemcpy(data_rgb_mem_buffer, image,
                       sizeof(uchar3) * height * width, cudaMemcpyDeviceToHost);
            image_ = cv::Mat(height, width, CV_8UC3, data_rgb_mem_buffer);
        }
        // 如果掩码数据非空，则从GPU拷贝到CPU的内存缓冲区，并创建对应的CV::Mat对象
        if (mask != nullptr) {
            cudaMemcpy(data_mask_mem_buffer, mask,
                       sizeof(uchar) * height * width, cudaMemcpyDeviceToHost);
            mask_ = cv::Mat(height, width, CV_8UC1, data_mask_mem_buffer);
        }
        return true;
    }
};

struct CylinderGPU_stilib {
    float* rotation;
    float* center;
    float r;
    float *global_theta, *global_phi;
    int2* boundary_pixel;
    int offset[4] = {0, 0, 0, 0};

    CylinderGPU_stilib() { }

    CylinderGPU_stilib(float* rotation_, float* center_, float r_) {
        rotation = rotation_;
        center   = center_;
        r        = r_;
        cudaMalloc((void**)&global_theta, sizeof(float) * 2);
        cudaMalloc((void**)&global_phi, sizeof(float) * 2);

        float min_theta = -3.141592653 / 180.0 * 80 + 0.001,
              max_theta = 3.141592653 / 180.0 * 80 - 0.001,
              min_phi   = -3.141592653 / 180.0 * 80,
              max_phi   = 3.141592653 / 180.0 * 80;
        cudaMemcpy(global_theta, &min_theta, sizeof(float) * 1,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(global_theta + 1, &max_theta, sizeof(float) * 1,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(global_phi, &min_phi, sizeof(float) * 1,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(global_phi + 1, &max_phi, sizeof(float) * 1,
                   cudaMemcpyHostToDevice);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
    }

    // 获取中心坐标
    inline __device__ float3 getCenter() {
        return make_float3(center[0], center[1], center[2]);
    }

    // 将一个三维向量 v 通过一个旋转矩阵的逆矩阵进行旋转
    inline __device__ float3 rotateVector_inv(float3 v) {
        return make_float3(
            rotation[0] * v.x + rotation[3] * v.y + rotation[6] * v.z,
            rotation[1] * v.x + rotation[4] * v.y + rotation[7] * v.z,
            rotation[2] * v.x + rotation[5] * v.y + rotation[8] * v.z);
    }
};

struct CylinderImageGPU_stilib {
    uchar3* image = nullptr;
    uchar* mask   = nullptr;
    int* uv;
    int height, width;

    CylinderImageGPU_stilib() { }

    CylinderImageGPU_stilib(int height_, int width_) {
        height = height_;
        width  = width_;
        cudaMalloc((void**)&image, sizeof(uchar3) * height * width);
        cudaMalloc((void**)&mask, sizeof(uchar) * height * width);
        cudaMalloc((void**)&uv, sizeof(int) * 2);
    }
};

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