#ifndef __PROJECT__
#define __PROJECT__

#include "cuda_runtime.h"
#include "cylinder_stitcher.cuh"
#include <cuda.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <vector>

#define CLOSE_ZERO 1.0e-6
// #define STEP 0.002f

bool projToCylinderImage_cuda(ViewGPU_stilib view,
                              CylinderImageGPU_stilib& cyl_image,
                              CylinderGPU_stilib& cylinder, int cyl_image_width,
                              int cyl_image_height);

#endif