#pragma once

//#define OUTPUT_CYL_IMAGE

#include "helper_cuda.h"
#include "math_utils.h"
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

    inline __device__ float3 rotateVector(float3 v) {
        return make_float3(R[0] * v.x + R[1] * v.y + R[2] * v.z,
                           R[3] * v.x + R[4] * v.y + R[5] * v.z,
                           R[6] * v.x + R[7] * v.y + R[8] * v.z);
    }

    inline __device__ float3 getCenter() {
        return make_float3(C[0], C[1], C[2]);
    }

    inline __device__ float2 projWorldToPixel(float3 x) {
        x.x -= C[0];
        x.y -= C[1];
        x.z -= C[2];
        float3 cam = this->rotateVector(x);

        if (cam.z < 0)
            return make_float2(-1, -1);
        return make_float2(cam.x * fx / cam.z + cx, cam.y * fy / cam.z + cy);
    }

    inline __device__ float2 projWorldToPixelFishEye(float3 x) {
        x.x -= C[0];
        x.y -= C[1];
        x.z -= C[2];
        float3 cam = this->rotateVector(x);

        if (cam.z < 0)
            return make_float2(-1, -1);

        float xx = cam.x / cam.z, yy = cam.y / cam.z;

        float r     = sqrtf(xx * xx + yy * yy);
        float theta = atan(r);

        float theta2 = theta * theta, theta4 = theta2 * theta2,
              theta6 = theta4 * theta2, theta8 = theta4 * theta4;
        float theta_d =
            theta * (1 + d0 * theta2 + d1 * theta4 + d2 * theta6 + d3 * theta8);

        float scale = (r == 0) ? 1.0 : theta_d / r;
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

    bool toCPU(cv::Mat& image_, cv::Mat& mask_) {
        if (data_rgb_mem_buffer == nullptr)
            cudaHostAlloc((void**)&data_rgb_mem_buffer,
                          sizeof(uchar3) * height * width,
                          cudaHostAllocDefault);
        if (data_mask_mem_buffer == nullptr)
            cudaHostAlloc((void**)&data_mask_mem_buffer,
                          sizeof(uchar) * height * width, cudaHostAllocDefault);

        if (image != nullptr) {
            cudaMemcpy(data_rgb_mem_buffer, image,
                       sizeof(uchar3) * height * width, cudaMemcpyDeviceToHost);
            image_ = cv::Mat(height, width, CV_8UC3, data_rgb_mem_buffer);
        }
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

    inline __device__ float3 getCenter() {
        return make_float3(center[0], center[1], center[2]);
    }

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

void ConvertRGBAF2RGBU_host(float* img_rgba, uchar3* img_rgb, int width,
                            int height, int grid, int block);

__global__ void ConvertRGBAF2RGBU(float* img_rgba, uchar3* img_rgb, int width,
                                  int height);