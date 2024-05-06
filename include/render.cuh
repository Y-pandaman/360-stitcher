#ifndef __RENDER__
#define __RENDER__

#include "cuda_runtime.h"
#include "cylinder_stitcher.cuh"
#include "project_to_cylinder.cuh"
#include <cuda.h>
#include <iostream>
#include <thrust/extrema.h>
#include <vector>

__host__ void BlendExtraViewToScreen_cuda(uchar3* dst_cyl_img,
                                          uchar3* src_cyl_img, int width,
                                          int height, float w);

#endif