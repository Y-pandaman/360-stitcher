#ifndef __RENDER__
#define __RENDER__

#include "cuda_runtime.h"
#include "cylinder_stitcher.cuh"
#include "project_to_cylinder.cuh"
#include <cuda.h>
#include <iostream>
#include <thrust/extrema.h>
#include <vector>

static inline __device__ void get_intrinsic(float& fx, float& fy, float& cx,
                                            float& cy, float image_theta_range,
                                            float image_phi_range, int height,
                                            int width) {
    fx = (width / 2.0f) / tanf(image_theta_range / 2.0f);
    fy = height / image_phi_range;
    cx = (width - 1.0f) / 2.0f;
    cy = (height - 1.0f) / 2.0f;
}

static inline __device__ float3 rotateVector_inv(float3 X_axis, float3 Y_axis,
                                                 float3 Z_axis, float3 v) {
    return make_float3(X_axis.x * v.x + Y_axis.x * v.y + Z_axis.x * v.z,
                       X_axis.y * v.x + Y_axis.y * v.y + Z_axis.y * v.z,
                       X_axis.z * v.x + Y_axis.z * v.y + Z_axis.z * v.z);
}

static inline __device__ float3 rotateVector(float3 X_axis, float3 Y_axis,
                                             float3 Z_axis, float3 v) {
    return make_float3(X_axis.x * v.x + X_axis.y * v.y + X_axis.z * v.z,
                       Y_axis.x * v.x + Y_axis.y * v.y + Y_axis.z * v.z,
                       Z_axis.x * v.x + Z_axis.y * v.y + Z_axis.z * v.z);
}

__host__ void BlendExtraViewToScreen_cuda(uchar3* dst_cyl_img,
                                          uchar3* src_cyl_img, int width,
                                          int height, float w);

// __host__ void RenderToScreen_cuda(std::vector<float4*>& novel_view_intrins,
//                                   std::vector<float3*>& novel_view_extrin_Rs,
//                                   CylinderImageGPU_stilib cyl_img,
//                                   std::vector<uchar3*> novel_images,
//                                   CylinderGPU_stilib cylinder, float3 origin,
//                                   int novel_images_num);

#endif