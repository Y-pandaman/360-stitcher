/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:52:03
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-06 16:17:08
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#ifndef __PROJECT__
#define __PROJECT__

#include "cuda_runtime.h"
#include "cylinder_stitcher.cuh"
#include <cuda.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <vector>

bool projToCylinderImage_cuda(ViewGPU_stilib view,
                              CylinderImageGPU_stilib& cyl_image,
                              CylinderGPU_stilib& cylinder, int cyl_image_width,
                              int cyl_image_height);

#endif