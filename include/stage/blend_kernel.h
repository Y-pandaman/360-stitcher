#ifndef __BLEND__
#define __BLEND__

#include "common/cylinder_stitcher.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include <vector>

__host__ void MultiBandBlend_cuda(std::vector<CylinderImageGPU> cylImages,
                                  std::vector<uchar*> seam_masks);

#endif