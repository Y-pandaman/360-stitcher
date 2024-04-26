#ifndef __BLEND__
#define __BLEND__

#include "cuda_runtime.h"
#include "cylinder_stitcher.h"
#include <cuda.h>
#include <iostream>
#include <thrust/extrema.h>
#include <vector>

__device__ static inline float3 operator*(float b, const uchar3 a) {
    return make_float3((float)a.x * b, (float)a.y * b, (float)a.z * b);
}

__device__ static inline float3 operator*(float b, const short3 a) {
    return make_float3((float)a.x * b, (float)a.y * b, (float)a.z * b);
}

__host__ void MultiBandBlend_cuda(std::vector<CylinderImageGPU> cylImages,
                                  std::vector<uchar*> seam_masks);

#endif