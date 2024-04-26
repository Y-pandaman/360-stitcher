#include "render.cuh"

__global__ void blend_extra_view_kernel(uchar3* dst_cyl_img,
                                        uchar3* src_cyl_img, int width,
                                        int height, float w) {
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (pixelIdx < width * height) {
        if (src_cyl_img[pixelIdx].x > 0 || src_cyl_img[pixelIdx].y > 0 ||
            src_cyl_img[pixelIdx].z > 0) {
            float w_1 =
                max(src_cyl_img[pixelIdx].x,
                    max(src_cyl_img[pixelIdx].y, src_cyl_img[pixelIdx].z)) /
                255.0 * w;
            float w_2               = 1.0 - w_1;
            dst_cyl_img[pixelIdx].x = (w_1 * (float)src_cyl_img[pixelIdx].x +
                                       w_2 * (float)dst_cyl_img[pixelIdx].x);
            dst_cyl_img[pixelIdx].y = (w_1 * (float)src_cyl_img[pixelIdx].y +
                                       w_2 * (float)dst_cyl_img[pixelIdx].y);
            dst_cyl_img[pixelIdx].z = (w_1 * (float)src_cyl_img[pixelIdx].z +
                                       w_2 * (float)dst_cyl_img[pixelIdx].z);
        }
    }
}

__host__ void BlendExtraViewToScreen_cuda(uchar3* dst_cyl_img,
                                          uchar3* src_cyl_img, int width,
                                          int height, float w) {
    int num_thread = 512;
    int num_block  = min(65535, (height * width + num_thread - 1) / num_thread);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    blend_extra_view_kernel<<<num_block, num_thread>>>(dst_cyl_img, src_cyl_img,
                                                       width, height, w);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}
