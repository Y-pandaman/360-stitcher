#include "cylinder_stitcher.cuh"

__global__ void ConvertRGBAF2RGBU(float* img_rgba, uchar3* img_rgb, int width,
                                  int height) {
    int pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;

    int max_idx = height * width;
    if (pixel_idx >= max_idx)
        return;

    int dst_row = pixel_idx / width, dst_col = pixel_idx % width;
    int src_row = height - dst_row - 1, src_col = dst_col;
    int src_pixel_idx    = (src_row * width + src_col) * 4;
    img_rgb[pixel_idx].x = img_rgba[src_pixel_idx] * 255;
    img_rgb[pixel_idx].y = img_rgba[src_pixel_idx + 1] * 255;
    img_rgb[pixel_idx].z = img_rgba[src_pixel_idx + 2] * 255;
}

void ConvertRGBAF2RGBU_host(float* img_rgba, uchar3* img_rgb, int width,
                            int height, int grid, int block) {
    ConvertRGBAF2RGBU<<<grid, block>>>(img_rgba, img_rgb, width, height);
}
