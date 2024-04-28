#include "render.cuh"

// 将一个源图像src_cyl_img按照给定的权重w混合到目标图像dst_cyl_img中
// __global__属性，表明这个函数可以在GPU上执行
__global__ void blend_extra_view_kernel(uchar3* dst_cyl_img,
                                        uchar3* src_cyl_img, int width,
                                        int height, float w) {
    // threadIdx.x是当前线程在块内的X坐标，blockIdx.x是当前块在网格中的X坐标，blockDim.x是每个块的线程数。
    // 计算当前线程处理的像素索引
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (pixelIdx < width * height) {
        // 检查源图像中当前像素的RGB三个通道中是否有非零值
        if (src_cyl_img[pixelIdx].x > 0 || src_cyl_img[pixelIdx].y > 0 ||
            src_cyl_img[pixelIdx].z > 0) {
            // 计算第一个权重w_1，它是源图像像素的RGB最大值除以255（因为图像数据类型是uchar3，范围是0-255），然后乘以混合权重w
            float w_1 =
                max(src_cyl_img[pixelIdx].x,
                    max(src_cyl_img[pixelIdx].y, src_cyl_img[pixelIdx].z)) /
                255.0 * w;
            // 计算第二个权重w_2，它是1减去w_1，用于平衡源图像和目标图像的像素值
            float w_2 = 1.0 - w_1;

            // 使用计算出的权重对目标图像的红色通道进行混合。将源图像的红色通道乘以w_1，目标图像的红色通道乘以w_2，然后相加。
            dst_cyl_img[pixelIdx].x = (w_1 * (float)src_cyl_img[pixelIdx].x +
                                       w_2 * (float)dst_cyl_img[pixelIdx].x);
            dst_cyl_img[pixelIdx].y = (w_1 * (float)src_cyl_img[pixelIdx].y +
                                       w_2 * (float)dst_cyl_img[pixelIdx].y);
            dst_cyl_img[pixelIdx].z = (w_1 * (float)src_cyl_img[pixelIdx].z +
                                       w_2 * (float)dst_cyl_img[pixelIdx].z);
        }
    }
}

// __host__属性，表明这是一个可以在CPU上执行的函数，但它将调用一个在GPU上执行的CUDA内核
__host__ void BlendExtraViewToScreen_cuda(uchar3* dst_cyl_img,
                                          uchar3* src_cyl_img, int width,
                                          int height, float w) {
    int num_thread = 512;   // 设置每个CUDA块（block）中的线程数为512
    // 计算需要多少个CUDA块来处理整个图像
    int num_block = min(65535, (height * width + num_thread - 1) / num_thread);

    checkCudaErrors(cudaDeviceSynchronize());   // 等待GPU完成所有先前排队的工作
    checkCudaErrors(cudaGetLastError());   // 检查CUDA操作是否有错误发生
    // <<<num_block,num_thread>>>定义了内核执行的网格（grid）和块（block）的配置
    // 内核将对目标图像dst_cyl_img和源图像src_cyl_img进行混合，混合的权重为w。
    blend_extra_view_kernel<<<num_block, num_thread>>>(dst_cyl_img, src_cyl_img,
                                                       width, height, w);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}
