#include "blend.h"

__device__ __constant__ float GauKernel[25] = {
    0.0039, 0.0156, 0.0234, 0.0156, 0.0039, 0.0156, 0.0625, 0.0938, 0.0625,
    0.0156, 0.0234, 0.0938, 0.1406, 0.0938, 0.0234, 0.0156, 0.0625, 0.0938,
    0.0625, 0.0156, 0.0039, 0.0156, 0.0234, 0.0156, 0.0039};

/**
 * @brief 对给定颜色通道进行加权乘积计算。
 *
 * 该模板函数针对一个特定的颜色通道，将源图像的像素值与给定权重相乘，
 * 并累加到对应的颜色值和权重和中。
 *
 * @tparam CHANNEL 颜色通道的数量。
 * @param color 指向存储颜色值的数组的指针。
 * @param src 指向源图像像素值的数组的指针。
 * @param aw 参与累加的权重总和的引用。
 * @param w 当前像素的权重。
 */
template <int CHANNEL>
__device__ static inline void weight_product(float* color, short* src,
                                             float& aw, float w) {
#pragma unroll
    // 遍历颜色通道，对每个通道的色彩值进行加权累加
    for (int i = 0; i < CHANNEL; i++) {
        color[i] = color[i] + (float)src[i] * w;
    }
    // 累加权重
    aw += w;
}

/**
 * @brief 根据一组掩码和源数据转换回目标数据。
 *
 * 该内核函数用于将经过特定处理的短期整型像素数据转换回无符号字符型像素数据，
 * 同时应用一组掩码来决定哪些像素应该被处理。处理过程中会将源像素的值根据掩码
 * 指定的条件转换到0到255的范围内。
 *
 * @param src 源像素数据的指针，类型为short3*，表示待转换的短期整型像素。
 * @param dst
 * 目标像素数据的指针，类型为uchar3*，表示转换后存储的无符号字符型像素。
 * @param mask0 到 mask5
 * 掩码数组，类型为uchar*，每个掩码数组用于指定哪些像素应被处理。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 */
__global__ void convertBack_kernel_6(short3* src, uchar3* dst, uchar* mask0,
                                     uchar* mask1, uchar* mask2, uchar* mask3,
                                     uchar* mask4, uchar* mask5, int height,
                                     int width) {
    // 计算当前线程处理的像素索引和所有线程的总数。
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 遍历所有像素，进行转换处理。
    while (pixelIdx < totalPixel) {
        // 根据掩码计算当前像素是否应被处理。
        int remian = mask0[pixelIdx] / 255 | mask1[pixelIdx] / 255 |
                     mask2[pixelIdx] / 255 | mask3[pixelIdx] / 255 |
                     mask4[pixelIdx] / 255 | mask5[pixelIdx] / 255;

        // 对应像素的转换处理，将短期整型值转换到0-255范围内。
        dst[pixelIdx].x = remian *
                          clamp((float)src[pixelIdx].x, 0.0f, 32767.0f) /
                          32767.0f * 255.0f;
        dst[pixelIdx].y = remian *
                          clamp((float)src[pixelIdx].y, 0.0f, 32767.0f) /
                          32767.0f * 255.0f;
        dst[pixelIdx].z = remian *
                          clamp((float)src[pixelIdx].z, 0.0f, 32767.0f) /
                          32767.0f * 255.0f;

        // 更新处理的像素索引。
        pixelIdx += total_thread;
    }
}

/**
 * @brief 根据给定的mask将short3数据转换为uchar3格式，并应用到指定的目标数组中。
 *
 * @param src 源数据数组，类型为short3，表示需要转换的原始图像数据。
 * @param dst 目标数据数组，类型为uchar3，用于存储转换后的图像数据。
 * @param mask0 到mask3 分别为4个掩码数组，类型为uchar，用于选择像素是否被处理。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 此函数通过线程在GPU上并行执行数据转换。每个线程处理一个像素点。
 * 首先，通过mask确定像素是否被处理（如果任何一个mask的值大于等于255，则该像素被处理）。
 * 然后，将src中的short值转换为uchar值，并根据mask确定是否应用该转换后的值到dst中。
 * 转换过程中使用了clamp函数确保值在有效范围内（0到32767）。
 */
__global__ void convertBack_kernel_4(short3* src, uchar3* dst, uchar* mask0,
                                     uchar* mask1, uchar* mask2, uchar* mask3,
                                     int height, int width) {
    int pixelIdx =
        threadIdx.x + blockIdx.x * blockDim.x;   // 计算当前线程处理的像素索引
    int total_thread = blockDim.x * gridDim.x;   // 计算总线程数
    int totalPixel   = height * width;           // 计算图像总像素数

    while (pixelIdx < totalPixel) {   // 循环处理所有像素
        // 根据mask确定当前像素是否被处理
        int remian = mask0[pixelIdx] / 255 | mask1[pixelIdx] / 255 |
                     mask2[pixelIdx] / 255 | mask3[pixelIdx] / 255;

        // 对R、G、B通道分别进行转换处理并更新到dst中
        dst[pixelIdx].x = remian *
                          clamp((float)src[pixelIdx].x, 0.0f, 32767.0f) /
                          32767.0f * 255.0f;
        dst[pixelIdx].y = remian *
                          clamp((float)src[pixelIdx].y, 0.0f, 32767.0f) /
                          32767.0f * 255.0f;
        dst[pixelIdx].z = remian *
                          clamp((float)src[pixelIdx].z, 0.0f, 32767.0f) /
                          32767.0f * 255.0f;

        pixelIdx += total_thread;   // 更新处理的像素索引
    }
}

/**
 * 模板函数，用于将给定通道数的源图像数据转换为指定数据类型，并应用缩放因子和掩码。
 *
 * @tparam CHANNEL 源图像的通道数。
 * @param src 源图像数据的指针。
 * @param mask 应用到源图像的掩码，用于指定某些像素是否要被处理。
 * @param dst 转换后图像数据的指针。
 * @param scale 缩放因子，用于调整源图像数据的值。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 函数在GPU上并行执行，将源图像数据（ uchar 类型）转换为 short
 * 类型，并应用缩放。
 * 如果源图像为三通道且掩码对应像素值为0，则将该像素所有通道的值设置为-32768。
 */
template <int CHANNEL>
__global__ void converTo_kernel(uchar* src, uchar* mask, short* dst,
                                float scale, int height, int width) {
    // 每个线程处理的像素索引
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // 总线程数
    int total_thread = blockDim.x * gridDim.x;
    // 图像总像素数
    int totalPixel = height * width;

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
#pragma unroll
        // 遍历当前像素的所有通道，应用缩放因子
        for (int i = 0; i < CHANNEL; i++) {
            dst[CHANNEL * pixelIdx + i] =
                (float)src[CHANNEL * pixelIdx + i] * scale;
        }

        // 如果是三通道图像且掩码值为0，则将像素值设置为-32768
        if (CHANNEL == 3 && mask[pixelIdx] == 0) {
#pragma unroll
            for (int i = 0; i < CHANNEL; i++) {
                dst[CHANNEL * pixelIdx + i] = -32768;
            }
        }

        // 更新处理的像素索引
        pixelIdx += total_thread;
    }
}

/*
ALGORITHM description
The function performs the downsampling step of the Gaussian pyramid
construction. First, it convolves the source image with the kernel: GauKernel
Then, it downsamples the image by rejecting even rows and columns.

Each thread is responsible for each pixel in high level (low resolution).
*/
/**
 * @brief 下采样金字塔 kernel 函数
 *
 * 该模板函数用于对给定的图像进行下采样，创建一个低分辨率的图像版本。
 * 它通过应用高斯滤波器来平均相邻像素，从而降低图像的分辨率。
 *
 * @tparam CHANNEL 图像的通道数
 * @param src 输入图像的指针（短整型）
 * @param dst 输出图像的指针（短整型）
 * @param height 输入图像的高度
 * @param width 输入图像的宽度
 *
 * 注意：该函数假设输入图像的尺寸是可被 2 整除的。
 */
template <int CHANNEL>
__global__ void PyrDown_kernel(short* src, short* dst, int height, int width) {
    // printf("fo02 ");
    // 计算当前线程处理的像素索引和总计数线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
        // 计算输入像素的坐标和索引
        int src_x = 2 * (pixelIdx % width);
        int src_y = 2 * (pixelIdx / width);

        // 初始化颜色数组和权重
        float color[CHANNEL];
        for (int i = 0; i < CHANNEL; i++)
            color[i] = 0;
        float weight = 0.0f;

        // 应用 5x5 高斯滤波器
#pragma unroll
        for (int i = 0; i < 5; i++) {
#pragma unroll
            for (int j = 0; j < 5; j++) {
                int u = src_x + j - 2;
                int v = src_y + i - 2;

                // 忽略越界的像素
                if (u < 0 || v < 0 || u >= width * 2 || v >= height * 2)
                    continue;

                // 忽略值为 -32768 的无效像素
                if (src[CHANNEL * (v * 2 * width + u)] == -32768)
                    continue;

                // 计算像素的加权颜色值
                weight_product<CHANNEL>(color,
                                        src + CHANNEL * (v * 2 * width + u),
                                        weight, GauKernel[i * 5 + j]);
            }
        }

        // 根据权重平均颜色值，并存储到输出图像中
        if (weight == 0) {
            // 如果权重为 0，则将像素设置为无效值 -32768
            for (int i = 0; i < CHANNEL; i++) {
                dst[CHANNEL * pixelIdx + i] = -32768;
            }
        } else {
            // 如果权重不为 0，则将加权颜色值存储到输出图像中
            for (int i = 0; i < CHANNEL; i++) {
                color[i]                    = color[i] / weight;
                dst[CHANNEL * pixelIdx + i] = (short)color[i];
            }
        }

        // 更新处理的像素索引
        pixelIdx += total_thread;
    }
}

/*
ALGORITHM description
The image is first upsized to twice the original in each dimension, with the
new (even) rows filled with 0s. Thereafter, a convolution is performed with
Gaussian filter to approximate the value of the "missing" pixels.

This filter is also normalized to 4, rather rhan 1. (because inserted rows have
0s)

Each thread is responsible for each pixel in low level (high resolution).
*/
/**
 * @brief PyrUp操作的CUDA内核函数，用于将输入图像放大两倍。
 *
 * @tparam SIGN 一个符号标志，用于控制放大操作是增加像素值还是减去像素值。
 * @param src 输入图像的短整型三维数组。
 * @param dst 输出图像的短整型三维数组。
 * @param height 输入图像的高度。
 * @param width 输入图像的宽度。
 *
 * 该函数使用了5x5的高斯核进行放大操作。每个输出像素由输入图像中附近25个像素经过加权平均得到。
 * 根据SIGN的值，可以实现放大像素值（SIGN=1）或减小像素值（SIGN=-1）的效果。
 */
template <int SIGN>
__global__ void PyrUp_kernel(short3* src, short3* dst, int height, int width) {
    // 计算当前线程处理的像素索引和总计线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
        // 计算当前处理的像素的x和y坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 初始化颜色向量和权重
        float3 color = make_float3(0, 0, 0);
        float weight = 0;

        // 使用双重循环应用5x5高斯核进行加权平均
#pragma unroll
        for (int i = 0; i < 5; i++) {
#pragma unroll
            for (int j = 0; j < 5; j++) {
                int u = x + j - 2;
                int v = y + i - 2;

                // 忽略越界的像素
                if (u < 0 || v < 0 || u >= width || v >= height)
                    continue;

                // 忽略无效像素（值为-32768）
                if (src[v / 2 * width / 2 + u / 2].x == -32768)
                    continue;

                // 计算当前像素的加权贡献
                if (u % 2 == 0 && v % 2 == 0) {
                    weight_product<3>(
                        &color.x, (short*)(src + (v / 2 * width / 2 + u / 2)),
                        weight, 4 * GauKernel[i * 5 + j]);
                }
            }
        }

        // 如果权重不为0，则更新像素值
        if (weight != 0) {
            // 根据权重调整颜色值
            color = color / weight;

            // 根据SIGN标志调整像素值
            color.x = (float)dst[pixelIdx].x + SIGN * color.x;
            color.y = (float)dst[pixelIdx].y + SIGN * color.y;
            color.z = (float)dst[pixelIdx].z + SIGN * color.z;

            // 确保像素值在合法范围内
            color = clamp(color, -32768.0f, 32767.0f);
        }

        // 更新输出图像的像素值
        dst[pixelIdx].x = color.x;
        dst[pixelIdx].y = color.y;
        dst[pixelIdx].z = color.z;

        // 移动到下一个像素
        pixelIdx += total_thread;
    }
}

// [NOTE] ******* 输出到I1 ***************
/**
 * 使用权重对四个图像进行融合的内核函数。
 *
 * @param I0 输入图像0的像素值（短整型三维向量）数组。
 * @param mask0 输入图像0的权重（短整型）数组。
 * @param I1 输入图像1的像素值（短整型三维向量）数组，也是输出图像的存储位置。
 * @param mask1 输入图像1的权重（短整型）数组。
 * @param I2 输入图像2的像素值（短整型三维向量）数组。
 * @param mask2 输入图像2的权重（短整型）数组。
 * @param I3 输入图像3的像素值（短整型三维向量）数组。
 * @param mask3 输入图像3的权重（短整型）数组。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 注意：此函数假设mask数组中的值在0到32767之间，且图像的维度为height * width。
 */
__global__ void WeightedBlend_kernel_4(short3* I0, short* mask0, short3* I1,
                                       short* mask1, short3* I2, short* mask2,
                                       short3* I3, short* mask3, int height,
                                       int width) {
    int pixelIdx =
        threadIdx.x + blockIdx.x * blockDim.x;   // 计算当前线程处理的像素索引
    int total_thread = blockDim.x * gridDim.x;   // 计算总线程数
    int totalPixel   = height * width;           // 计算图像总像素数

    // 遍历所有像素
    while (pixelIdx < totalPixel) {
        // 计算四个图像对应像素的权重
        float w0      = (float)mask0[pixelIdx] / 32767.0f;
        float w1      = (float)mask1[pixelIdx] / 32767.0f;
        float w2      = (float)mask2[pixelIdx] / 32767.0f;
        float w3      = (float)mask3[pixelIdx] / 32767.0f;
        float total_w = w0 + w1 + w2 + w3;   // 计算总权重

        if (total_w > 0) {   // 当总权重大于0时进行融合处理
            // 根据权重计算融合后的像素值
            I1[pixelIdx].x = (short)(((float)I0[pixelIdx].x * w0 +
                                      (float)I1[pixelIdx].x * w1 +
                                      (float)I2[pixelIdx].x * w2 +
                                      (float)I3[pixelIdx].x * w3) /
                                     total_w);
            I1[pixelIdx].y = (short)(((float)I0[pixelIdx].y * w0 +
                                      (float)I1[pixelIdx].y * w1 +
                                      (float)I2[pixelIdx].y * w2 +
                                      (float)I3[pixelIdx].y * w3) /
                                     total_w);
            I1[pixelIdx].z = (short)(((float)I0[pixelIdx].z * w0 +
                                      (float)I1[pixelIdx].z * w1 +
                                      (float)I2[pixelIdx].z * w2 +
                                      (float)I3[pixelIdx].z * w3) /
                                     total_w);
        } else {   // 当总权重为0时，将像素值设为0
            I1[pixelIdx].x = 0;
            I1[pixelIdx].y = 0;
            I1[pixelIdx].z = 0;
        }
        pixelIdx += total_thread;   // 更新处理的像素索引
    }
}

/**
 * 在GPU上执行加权融合操作的内核函数。
 *
 * @param I0 输入图像1的像素值（短整型三维向量）数组。
 * @param mask0 输入图像1的掩码（短整型）数组。
 * @param I1 输入图像2的像素值（短整型三维向量）数组，同时也是输出图像的数组。
 * @param mask1 输入图像2的掩码（短整型）数组。
 * @param I2 输入图像3的像素值（短整型三维向量）数组。
 * @param mask2 输入图像3的掩码（短整型）数组。
 * @param I3 输入图像4的像素值（短整型三维向量）数组。
 * @param mask3 输入图像4的掩码（短整型）数组。
 * @param I4 输入图像5的像素值（短整型三维向量）数组。
 * @param mask4 输入图像5的掩码（短整型）数组。
 * @param I5 输入图像6的像素值（短整型三维向量）数组。
 * @param mask5 输入图像6的掩码（短整型）数组。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 该函数通过对输入的六幅图像的像素值加权融合，生成输出图像。每幅图像的像素值由对应的掩码
 * 来调整权重。加权融合的结果存储在第二幅图像（I1）中。
 */
__global__ void WeightedBlend_kernel_6(short3* I0, short* mask0, short3* I1,
                                       short* mask1, short3* I2, short* mask2,
                                       short3* I3, short* mask3, short3* I4,
                                       short* mask4, short3* I5, short* mask5,
                                       int height, int width) {
    int pixelIdx =
        threadIdx.x + blockIdx.x * blockDim.x;   // 计算当前线程处理的像素索引
    int total_thread = blockDim.x * gridDim.x;   // 计算总线程数
    int totalPixel   = height * width;           // 计算图像总像素数

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
        // 计算每幅图像对应像素的权重
        float w0 = (float)mask0[pixelIdx] / 32767.0f;
        float w1 = (float)mask1[pixelIdx] / 32767.0f;
        float w2 = (float)mask2[pixelIdx] / 32767.0f;
        float w3 = (float)mask3[pixelIdx] / 32767.0f;
        float w4 = (float)mask4[pixelIdx] / 32767.0f;
        float w5 = (float)mask5[pixelIdx] / 32767.0f;

        float total_w = w0 + w1 + w2 + w3 + w4 + w5;   // 计算所有权重的总和

        if (total_w > 0) {   // 当总权重大于0时进行加权融合
            // 对R、G、B通道分别进行加权融合
            I1[pixelIdx].x = (short)(((float)I0[pixelIdx].x * w0 +
                                      (float)I1[pixelIdx].x * w1 +
                                      (float)I2[pixelIdx].x * w2 +
                                      (float)I3[pixelIdx].x * w3 +
                                      (float)I4[pixelIdx].x * w4 +
                                      (float)I5[pixelIdx].x * w5) /
                                     total_w);
            I1[pixelIdx].y = (short)(((float)I0[pixelIdx].y * w0 +
                                      (float)I1[pixelIdx].y * w1 +
                                      (float)I2[pixelIdx].y * w2 +
                                      (float)I3[pixelIdx].y * w3 +
                                      (float)I4[pixelIdx].y * w4 +
                                      (float)I5[pixelIdx].y * w5) /
                                     total_w);
            I1[pixelIdx].z = (short)(((float)I0[pixelIdx].z * w0 +
                                      (float)I1[pixelIdx].z * w1 +
                                      (float)I2[pixelIdx].z * w2 +
                                      (float)I3[pixelIdx].z * w3 +
                                      (float)I4[pixelIdx].z * w4 +
                                      (float)I5[pixelIdx].z * w5) /
                                     total_w);
        } else {   // 当总权重为0时，将像素值设为0
            I1[pixelIdx].x = 0;
            I1[pixelIdx].y = 0;
            I1[pixelIdx].z = 0;
        }

        pixelIdx += total_thread;   // 更新处理的像素索引
    }
}

/**
 * 生成拉普拉斯金字塔
 *
 * @param img 输入的圆柱图像，类型为CylinderImageGPU
 * @param seam_mask 缝隙掩码，指向uchar类型的指针
 * @param levels 金字塔的级别数
 * @return 返回一个包含各级别图像的std::vector<SeamImageGPU>
 */
std::vector<SeamImageGPU> LaplacianPyramid(CylinderImageGPU img,
                                           uchar* seam_mask, int levels) {
    std::vector<SeamImageGPU> pyr_imgs;   // 存储金字塔图像的容器

    int num_thread = 512;                         // 每个块的线程数
    int height = img.height, width = img.width;   // 输入图像的高度和宽度
    // 计算块的数量，以适应CUDA线程块的限制
    int num_block = min(65535, (height * width + num_thread - 1) / num_thread);

    // 分配GPU内存以存储当前图像和掩码
    short3* current_img;
    cudaMalloc((void**)&current_img, sizeof(short3) * height * width);
    short* current_mask;
    cudaMalloc((void**)&current_mask, sizeof(short) * height * width);

    // 将输入图像和掩码转换为短整型格式
    converTo_kernel<3><<<num_block, num_thread>>>(
        (uchar*)img.image, img.mask, (short*)current_img, 32767.0f / 255.0f,
        height, width);
    converTo_kernel<1><<<num_block, num_thread>>>(
        seam_mask, NULL, current_mask, 32767.0f / 255.0f, height, width);

    // 将转换后的图像和掩码封装成SeamImageGPU对象，加入到结果容器中
    pyr_imgs.emplace_back(
        SeamImageGPU(current_img, current_mask, height, width));

    // 从上往下构建金字塔的剩余级别
    for (int i = 1; i < levels; i++) {
        height /= 2;   // 更新下一级别的图像高度和宽度
        width /= 2;
        // 根据新的图像尺寸计算块的数量
        num_block = min(65535, (height * width + num_thread - 1) / num_thread);

        // 分配内存以存储下一级别的图像和掩码
        short3* _down_scale_img;
        cudaMalloc((void**)&_down_scale_img, sizeof(short3) * height * width);
        short* _down_scale_mask;
        cudaMalloc((void**)&_down_scale_mask, sizeof(short) * height * width);

        // 对当前图像和掩码进行下采样
        PyrDown_kernel<3><<<num_block, num_thread>>>(
            (short*)current_img, (short*)_down_scale_img, height, width);
        PyrDown_kernel<1><<<num_block, num_thread>>>(
            current_mask, _down_scale_mask, height, width);

        // 将下采样后的图像和掩码封装成SeamImageGPU对象，加入到结果容器中
        pyr_imgs.emplace_back(
            SeamImageGPU(_down_scale_img, _down_scale_mask, height, width));

        // 更新当前图像和掩码指针，为下一次下采样做准备
        current_img  = _down_scale_img;
        current_mask = _down_scale_mask;
    }

    // 从下往上构建逆金字塔，除了最后一级
    for (int i = 0; i < levels - 1; i++) {
        // 根据当前级别图像的尺寸计算块的数量
        int num_block = min(
            65535, (pyr_imgs[i].height * pyr_imgs[i].width + num_thread - 1) /
                       num_thread);
        // 执行上采样操作
        PyrUp_kernel<-1><<<num_block, num_thread>>>(
            pyr_imgs[i + 1].image, pyr_imgs[i].image, pyr_imgs[i].height,
            pyr_imgs[i].width);
    }

    return pyr_imgs;   // 返回构建好的拉普拉斯金字塔图像容器
}

/**
 * 合并图像金字塔中的图像，通过加权混合生成一个新的图像金字塔。
 *
 * @param pyr_imgs
 * 一个包含多个图像金字塔的二维向量，每个图像金字塔又是一个包含多层图像的向量。
 *                 每层图像由SeamImageGPU结构体表示，包含图像数据和对应的掩码。
 * @return
 * 返回一个包含合并后图像的金字塔的向量，每个图像由SeamImageGPU结构体表示。
 */
std::vector<SeamImageGPU>
BlendPyramid(std::vector<std::vector<SeamImageGPU>>& pyr_imgs) {
    int num_thread = 512;   // 定义每个块中线程的数量

    std::vector<SeamImageGPU> blendedPyramid;   // 存储合并后的图像金字塔
    // 遍历图像金字塔的每一层
    for (uint64_t i = 0; i < pyr_imgs[0].size(); i++) {
        int num_block =   // 计算执行混合操作所需块的数量
            min(65535, (pyr_imgs[0][i].height * pyr_imgs[0][i].width +
                        num_thread - 1) /
                           num_thread);

        // 如果提供了6个图像金字塔，则使用6个图像的混合kernel
        if (pyr_imgs.size() == 6) {
            WeightedBlend_kernel_6<<<num_block, num_thread>>>(
                pyr_imgs[0][i].image, pyr_imgs[0][i].mask, pyr_imgs[1][i].image,
                pyr_imgs[1][i].mask, pyr_imgs[2][i].image, pyr_imgs[2][i].mask,
                pyr_imgs[3][i].image, pyr_imgs[3][i].mask, pyr_imgs[4][i].image,
                pyr_imgs[4][i].mask, pyr_imgs[5][i].image, pyr_imgs[5][i].mask,
                pyr_imgs[0][i].height, pyr_imgs[0][i].width);
        }
        // 如果提供了4个图像金字塔，则使用4个图像的混合kernel
        else if (pyr_imgs.size() == 4) {
            WeightedBlend_kernel_4<<<num_block, num_thread>>>(
                pyr_imgs[0][i].image, pyr_imgs[0][i].mask, pyr_imgs[1][i].image,
                pyr_imgs[1][i].mask, pyr_imgs[2][i].image, pyr_imgs[2][i].mask,
                pyr_imgs[3][i].image, pyr_imgs[3][i].mask,
                pyr_imgs[0][i].height, pyr_imgs[0][i].width);
        }

        blendedPyramid.emplace_back(SeamImageGPU(
            pyr_imgs[1][i].image, NULL, pyr_imgs[1][i].height,
            pyr_imgs[1][i].width));   // 将混合后的图像添加到结果金字塔中
    }
    return blendedPyramid;
}

/**
 * 将金字塔图像合并回原始尺寸。
 * 此函数主要用于将经过融合处理的图像金字塔合并回其原始尺寸，
 * 并根据需要将结果转换回圆柱形图像格式。
 *
 * @param blendedPyramid 包含融合后图像的金字塔。从底部到顶部，尺寸逐渐减小。
 * @param cylImages 圆柱形图像数组。根据需要，可能包含4或6个图像及其掩码。
 */

void CollapsePyramid(std::vector<SeamImageGPU> blendedPyramid,
                     std::vector<CylinderImageGPU> cylImages) {
    int num_thread = 512, num_block = 0;

    // 从金字塔的倒数第二层开始，逐层向上进行图像放大融合。
    for (int i = blendedPyramid.size() - 2; i >= 0; i--) {
        // 根据当前层的图像大小计算线程块和块数。
        num_block =
            min(65535, (blendedPyramid[i].height * blendedPyramid[i].width +
                        num_thread - 1) /
                           num_thread);
        // 使用PyrUp_kernel核函数将图像从下一层放大到当前层。
        PyrUp_kernel<1><<<num_block, num_thread>>>(
            blendedPyramid[i + 1].image, blendedPyramid[i].image,
            blendedPyramid[i].height, blendedPyramid[i].width);
    }

    // 根据输入的图像数量，选择合适的转换方法。
    if (cylImages.size() == 6) {
        // 如果有6个图像，则使用完整的转换核函数进行转换。
        convertBack_kernel_6<<<num_block, num_thread>>>(
            blendedPyramid[0].image, cylImages[1].image, cylImages[0].mask,
            cylImages[1].mask, cylImages[2].mask, cylImages[3].mask,
            cylImages[4].mask, cylImages[5].mask, blendedPyramid[0].height,
            blendedPyramid[0].width);
    } else if (cylImages.size() == 4) {
        // 如果只有4个图像，则使用简化版的转换核函数。
        convertBack_kernel_4<<<num_block, num_thread>>>(
            blendedPyramid[0].image, cylImages[1].image, cylImages[0].mask,
            cylImages[1].mask, cylImages[2].mask, cylImages[3].mask,
            blendedPyramid[0].height, blendedPyramid[0].width);
    }
}

/**
 * 在CUDA设备上执行多带混合操作。
 *
 * @param cylImages 一个包含多个CylinderImageGPU对象的向量，表示待处理的图像。
 * @param seam_masks
 * 一个包含指向每个图像缝合掩膜的指针的向量，用于指定缝合路径。
 */
__host__ void MultiBandBlend_cuda(std::vector<CylinderImageGPU> cylImages,
                                  std::vector<uchar*> seam_masks) {
    int levels = 5;   // 定义金字塔的层数

    // 为每张图像构建拉普拉斯金字塔
    std::vector<std::vector<SeamImageGPU>> pyr_imgs(cylImages.size());
    for (uint64_t i = 0; i < cylImages.size(); i++) {
        pyr_imgs[i] = LaplacianPyramid(cylImages[i], seam_masks[i], levels);
    }

    // 混合所有图像的金字塔
    std::vector<SeamImageGPU> blendedPyramid = BlendPyramid(pyr_imgs);

    // 合并金字塔回原始图像
    CollapsePyramid(blendedPyramid, cylImages);

    // 清理金字塔数据，释放内存
    for (uint64_t i = 0; i < pyr_imgs.size(); i++) {
        for (uint64_t j = 0; j < pyr_imgs[i].size(); j++) {
            pyr_imgs[i][j].clear();
        }
    }
}