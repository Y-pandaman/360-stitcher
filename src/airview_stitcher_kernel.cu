#include "airview_stitcher.h"

__device__ __constant__ uchar3 BBOX_COLOR   = {50, 205, 50};
__device__ __constant__ uchar3 CIRCLE_COLOR = {255, 144, 30};

// ================= DEVICE ==================================

/**
 * 实现二维双线性插值。
 *
 * @param src 输入图像指针，类型为T，存储的是像素值。
 * @param dst 输出图像指针，类型为T，存储插值后的像素值。
 * @param h 输入图像的高度。
 * @param w 输入图像的宽度。
 * @param pixel
 * 需要插值的像素点坐标，类型为float2，x和y分量分别为像素点的x和y坐标。
 * @note 该函数适用于设备端（__device__），且为内联函数（inline）。
 *
 * 对于给定的像素点坐标，该函数计算其在输入图像上的双线性插值像素值，
 * 并将结果存储在输出图像的相应位置。
 * 当像素点坐标超出图像边界时，将输出全零。
 */
template <typename T, int CHANNEL>
static inline __device__ void Bilinear(T* src, T* dst, int h, int w,
                                       float2 pixel) {
    // 检查像素点是否在图像边界内
    if (pixel.x < 1 || pixel.x >= w - 1 || pixel.y < 1 || pixel.y >= h - 1) {
#pragma unroll
        // 如果像素点在边界外，将输出通道的所有元素设为0
        for (int i = 0; i < CHANNEL; i++) {
            dst[i] = 0;
        }
        return;
    }

    // 计算四个邻近像素的坐标
    int x0 = (int)pixel.x;
    int y0 = (int)pixel.y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    // 计算像素点在小数部分的偏移
    float x = pixel.x - x0;
    float y = pixel.y - y0;

    // 计算四个邻近像素的索引
    int idx00 = x0 + y0 * w;
    int idx01 = idx00 + w;
    int idx10 = idx00 + 1;
    int idx11 = idx01 + 1;

#pragma unroll
    // 对每个通道进行双线性插值计算
    for (int i = 0; i < CHANNEL; i++) {
        dst[i] = (T)((float)src[idx00 * CHANNEL + i] * (1.0f - x) * (1.0f - y) +
                     (float)src[idx01 * CHANNEL + i] * (1.0f - x) * y +
                     (float)src[idx10 * CHANNEL + i] * x * (1.0f - y) +
                     (float)src[idx11 * CHANNEL + i] * x * y);
    }
}

/**
 * @brief 对给定通道内的颜色值与权重进行乘积累加。
 *
 * 该函数针对一个像素的多个通道（如RGB或RGBA），将每个通道的值与给定权重相乘，并将结果累加到对应的颜色值中。
 * 同时，还对总权重进行累加，以便于后续可能的归一化或其他操作。
 *
 * @tparam T 数据类型，模板参数，代表像素数据类型（如float，int等）。
 * @tparam CHANNEL
 * 常量整型，模板参数，指定像素的通道数（如3对应RGB，4对应RGBA）。
 * @param color 指向存储最终颜色值的数组的指针，数组长度为CHANNEL。
 * @param src 指向原始颜色值的数组的指针，数组长度为CHANNEL。
 * @param aw 参考参数，用于累加权重，函数内会修改其值。
 * @param w 给定的权重值，用于与每个通道的颜色值相乘并累加。
 */
template <typename T, int CHANNEL>
__device__ static inline void weight_product(float* color, T* src, float& aw,
                                             float w) {
#pragma unroll
    // 遍历所有通道，将每个通道的值与权重相乘后累加到color数组中
    for (int i = 0; i < CHANNEL; i++) {
        color[i] = color[i] + (float)src[i] * w;
    }
    // 累加总权重
    aw += w;
}

__device__ static inline float point2line_dis(float3 L, float3 P) {
    return fabs(dot(L, P)) / sqrt(L.x * L.x + L.y * L.y);
}

// ================= Kernel ===================================
/**
 * 在GPU上执行的计算网格核心函数。
 *
 * @param H 一个包含9个浮点数的数组，用于表示仿射变换矩阵。
 * @param grid 一个float2类型的数组，用于存储变换后的像素点坐标。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 此函数通过应用仿射变换矩阵H，将图像像素点映射到一个新的网格上。
 * 对于每个像素点，计算其在新网格上的坐标，并根据是否在网格内来更新grid数组。
 * 如果像素点位于网格内，则计算其在新网格上的坐标；否则，标记为无效坐标。
 */
__global__ void compute_grid_kernel(float* H, float2* grid, int height,
                                    int width) {
    // 计算当前线程处理的像素索引和所有线程能够处理的像素总数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 遍历所有像素点，分配给多个线程以提高计算效率
    while (pixelIdx < totalPixel) {
        // 计算当前像素点的行列坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 应用仿射变换矩阵H计算像素点在新网格上的坐标
        float x_ = H[0] * x + H[1] * y + H[2];
        float y_ = H[3] * x + H[4] * y + H[5];
        float z_ = H[6] * x + H[7] * y + H[8];

        // 判断像素点是否位于新网格内，更新grid数组
        if (z_ < 0) {
            float u = x_ / z_;
            float v = y_ / z_;

            grid[pixelIdx] = make_float2(u, v);

        } else {
            // 如果像素点不在网格内，则标记为无效坐标
            grid[pixelIdx] = make_float2(-1.0f, -1.0f);
        }
        // 更新处理的像素索引，准备处理下一个像素
        pixelIdx += total_thread;
    }
}

/**
 * @brief 使用双线性插值从网格中采样源图像并存储到目标图像中。
 *
 * @tparam T 源图像和目标图像的数据类型。
 * @tparam CHANNEL 图像的通道数。
 * @param grid
 * 存储了图像像素位置的网格，类型为float2*，其中每个元素包含了像素在图像平面上的(x,
 * y)坐标。
 * @param src 源图像的数据指针。
 * @param dst 目标图像的数据指针。
 * @param src_h 源图像的高度。
 * @param src_w 源图像的宽度。
 * @param height 目标图像的高度。
 * @param width 目标图像的宽度。
 *
 * 核心思想是利用双线性插值算法，根据给定的网格点信息，从源图像中采样并插值计算出对应在目标图像中的像素值。
 * 该操作对于图像变换（如放缩、旋转等）中保持图像质量是非常有用的。
 */
template <typename T, int CHANNEL>
__global__ void grid_sample_kernel(float2* grid, T* src, T* dst, int src_h,
                                   int src_w, int height, int width) {
    // 计算当前线程处理的像素索引和总计线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 遍历所有像素，使用双线性插值计算并更新目标图像
    while (pixelIdx < totalPixel) {
        // 对当前像素进行双线性插值采样，并将结果存储到目标图像中
        Bilinear<T, CHANNEL>(src, dst + pixelIdx * CHANNEL, src_h, src_w,
                             grid[pixelIdx]);

        // 更新像素索引，准备处理下一个像素
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
 * 该模板函数用于对给定的图像进行下采样操作，生成下一层金字塔的图像。它采用5x5的高斯核进行滤波。
 *
 * @tparam CHANNEL 图像通道数
 * @param src 输入图像的指针
 * @param dst 输出图像的指针
 * @param height 输入图像的高度
 * @param width 输入图像的宽度
 */
template <int CHANNEL>
__global__ void PyrDown_kernel(short* src, short* dst, int height, int width) {
    // 计算当前线程处理的像素点索引和总计线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 循环处理所有像素点
    while (pixelIdx < totalPixel) {
        // 计算原始图像中对应当前像素点的坐标和索引
        int src_x        = 2 * (pixelIdx % width);
        int src_y        = 2 * (pixelIdx / width);
        int src_pixelIdx = src_y * (width * 2) + src_x;

        // 初始化颜色数组和权重
        float color[CHANNEL];
        for (int i = 0; i < CHANNEL; i++)
            color[i] = 0;
        float weight = 0.0f;

        // 使用5x5高斯核进行滤波处理
#pragma unroll
        for (int i = 0; i < 5; i++) {
#pragma unroll
            for (int j = 0; j < 5; j++) {
                int u = src_x + j - 2;
                int v = src_y + i - 2;

                // 跳过越界的像素和值为特殊标识的像素
                if (u < 0 || v < 0 || u >= width * 2 || v >= height * 2)
                    continue;

                if (src[CHANNEL * (v * 2 * width + u)] == -32768)
                    continue;

                // 计算像素点的加权颜色值
                weight_product<short, CHANNEL>(
                    color, src + CHANNEL * (v * 2 * width + u), weight,
                    GauKernel[i * 5 + j]);
            }
        }

        // 根据权重计算最终颜色值并写入输出图像
#pragma unroll
        if (weight == 0) {
            // 如果权重为0，则将像素值设置为特殊标识
            for (int i = 0; i < CHANNEL; i++) {
                dst[CHANNEL * pixelIdx + i] = -32768;
            }
        } else {
            // 根据权重平均颜色值并转换为short类型写入输出图像
            for (int i = 0; i < CHANNEL; i++) {
                color[i]                    = color[i] / weight;
                dst[CHANNEL * pixelIdx + i] = (short)color[i];
            }
        }

        // 更新处理的像素点索引
        pixelIdx += total_thread;
    }
}

/**
 * 在GPU上执行的内核函数，用于将short3型像素值转换回uchar3型像素值。
 *
 * @param src 指向short3型像素数据的指针（输入）。
 * @param dst 指向uchar3型像素数据的指针（输出）。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 此函数假设短期整型像素值的范围是-32768到32767，并将其转换为无符号字符型像素值的范围0到255。
 * 使用线程块和网格来并行处理整个图像。
 */
__global__ void convertBack_kernel(short3* src, uchar3* dst, int height,
                                   int width) {
    // gridDim.x：表示一个Grid里面Block块数量
    // blockDim.x：表示一个Block中x方向上Thread数量
    // blockIdx.x：表示当前Block在Grid中的索引
    // threadIdx.x：表示当前Thread在对应Block中的索引
    // 计算当前线程处理的像素索引
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // 计算所有线程的总数
    int total_thread = blockDim.x * gridDim.x;
    // 计算图像的总像素数
    int totalPixel = height * width;

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
        // 计算当前像素的x和y坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 将短期整型像素值转换为无符号字符型像素值
        dst[pixelIdx].x = (uchar)((float)src[pixelIdx].x / 32767.0f * 255.0f);
        dst[pixelIdx].y = (uchar)((float)src[pixelIdx].y / 32767.0f * 255.0f);
        dst[pixelIdx].z = (uchar)((float)src[pixelIdx].z / 32767.0f * 255.0f);

        // 更新处理的像素索引，准备处理下一个像素
        pixelIdx += total_thread;
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

/**
 * @brief 计算两个图像之间的差异。
 *
 * 该内核函数在GPU上运行，用于计算两个输入图像（img1和img2）的每个像素之间的差异，
 * 并将结果存储在diff数组中。差异是通过比较每个像素的RGB分量之和，并对差异进行归一化计算得到的。
 *
 * @param img1 输入图像1的像素数组，类型为uchar3，表示RGB颜色空间。
 * @param img2 输入图像2的像素数组，类型为uchar3，表示RGB颜色空间。
 * @param diff 存储两个图像像素差异的输出数组，类型为ushort。
 * @param height 输入图像的高度。
 * @param width 输入图像的宽度。
 */
__global__ void diff_kernel(uchar3* img1, uchar3* img2, ushort* diff,
                            int height, int width) {
    // 每个线程处理一个像素，计算线程索引并确定总线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;   // 图像总像素数

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
        // 从图像数组中读取当前像素
        uchar3 rgb1 = img1[pixelIdx];
        uchar3 rgb2 = img2[pixelIdx];

        // 计算两个像素的RGB值之和
        int sum1 = rgb1.x + rgb1.y + rgb1.z;
        int sum2 = rgb2.x + rgb2.y + rgb2.z;

        // 处理全黑或全白像素，将其差异值设为最大
        if (sum1 == 0 || sum2 == 0) {
            diff[pixelIdx] = 65535;
            pixelIdx += total_thread;
            continue;
        }

        // 将RGB值归一化到0-1范围，并计算两个像素的差异
        float3 c1 =
            make_float3((float)rgb1.x, (float)rgb1.y, (float)rgb1.z) / MASK_MAX;
        float3 c2 =
            make_float3((float)rgb2.x, (float)rgb2.y, (float)rgb2.z) / MASK_MAX;
        float3 d = fabs(c1 - c2);   // 计算每种颜色通道的差异

        // 计算综合差异并转换为 ushort 类型
        float base = 0.5;
        diff[pixelIdx] =
            (ushort)((d.x + d.y + d.z + base) / (3.0f + base) * 65535);

        // 移动到下一个像素
        pixelIdx += total_thread;
    }
}

/**
 * 在给定的图像差异数组中，根据提供的缝合路径更新像素的差异值。
 * 该更新仅对缝合路径上的像素且其差异值不为最大值有效。
 *
 * @param input_output_diff 输入输出的图像差异数组，其值将被更新。
 * @param seam 缝合路径的数组，标识了哪些像素位于缝合路径上。
 * @param delta_on_seam 在缝合路径上像素的差异值增量。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 */
__global__ void update_diff_use_seam_kernel(ushort* input_output_diff,
                                            uchar* seam, ushort delta_on_seam,
                                            int height, int width) {
    // 计算当前线程处理的像素索引
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // 计算所有线程的总数
    int total_thread = blockDim.x * gridDim.x;
    // 计算图像总像素数
    int totalPixel = height * width;

    // 遍历所有像素，更新差异值
    while (pixelIdx < totalPixel) {
        // 如果像素位于缝合路径上且差异值不是最大值，则将其差异值更新为0
        if (seam[pixelIdx] != 255 && input_output_diff[pixelIdx] != 65535)
            input_output_diff[pixelIdx] = 0;
        // 跳转到下一个像素
        pixelIdx += total_thread;
    }
}

/**
 * 在CUDA设备上执行的kernel函数，用于计算图像中指定区域像素的差异。
 *
 * @param diff 指向ushort类型的数组，用于存储计算得到的像素差异。
 * @param grid 指向float2类型的数组，表示图像的UV坐标网格。
 * @param scale 缩放因子，用于确定grid中每个像素点对应的实际图像像素位置。
 * @param x_min x轴上的最小区域边界。
 * @param y_min y轴上的最小区域边界。
 * @param x_max x轴上的最大区域边界。
 * @param y_max y轴上的最大区域边界。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 该函数遍历图像的所有像素，对于位于指定区域内的像素，将其在diff数组中的值设置为65535。
 */
__global__ void diff_person_kernel(ushort* diff, float2* grid, int scale,
                                   float x_min, float y_min, float x_max,
                                   float y_max, int height, int width) {
    // 计算当前线程处理的像素索引和总计线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;   // 图像总像素数

    // 遍历所有像素
    while (pixelIdx < totalPixel) {
        // 计算当前像素的行列坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 获取当前像素的UV坐标
        float2 uv = grid[y * scale * (width * scale) + x * scale];

        // 检查当前像素是否位于指定的区域范围内
        if (uv.x >= x_min && uv.x <= x_max && uv.y >= y_min && uv.y <= y_max) {
            // 如果是，则将差异值设置为65535
            diff[pixelIdx] = 65535;
        }

        // 更新像素索引，准备处理下一个像素
        pixelIdx += total_thread;
    }
}

/**
 * 在图像中绘制图标
 *
 * 该内核函数用于将一个图标绘制到给定的图像上。图标可以是任意大小，但会居中绘制在图像中心。
 * 如果图标的任何像素的alpha通道值大于128，它将覆盖图像中对应位置的像素。
 *
 * @param img 指向图像像素数据的指针，格式为uchar3（RGB）。
 * @param icon 指向图标像素数据的指针，格式为uchar4（RGBA）。
 * @param icon_h 图标的高度。
 * @param icon_w 图标的宽度。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 */
__global__ void draw_car_kernel(uchar3* img, uchar4* icon, int icon_h,
                                int icon_w, int height, int width) {
    // 计算当前线程处理的像素索引和线程总数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;   // 图像总像素数

    // 调整图标在图像中的中心位置
    // int cx = width / 2 + 15; // TODO: 会增加耗时
    // int cy = height / 2 + 15;
    int cx = width / 2;
    int cy = height / 2;
    // 计算图标半宽、半高
    int half_w = icon_w / 2;
    int half_h = icon_h / 2;

    // 计算图标在图像中的边界
    int min_x = cx - half_w;
    int min_y = cy - half_h;
    int max_x = min_x + icon_w;
    int max_y = min_y + icon_h;

    // 遍历图像中所有像素
    while (pixelIdx < totalPixel) {
        // 计算当前像素的行列坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 检查当前像素是否在图标范围内
        if (x >= min_x && x < max_x && y >= min_y && y < max_y) {
            // 获取当前像素的图标像素
            uchar4 rgba = icon[(y - min_y) * icon_w + (x - min_x)];

            // 如果图标像素的alpha值大于128，覆盖图像像素
            if (rgba.w > 128) {
                img[pixelIdx] = make_uchar3(rgba.x, rgba.y, rgba.z);
            }
        }

        // 更新处理的像素索引，准备处理下一个像素
        pixelIdx += total_thread;
    }
}

__global__ void draw_circle_kernel(uchar3* img, float2 center, float radius,
                                   int thickness, int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        float dis = sqrt((x - center.x) * (x - center.x) +
                         (y - center.y) * (y - center.y));
        dis       = fabs(dis - radius);

        if (dis <= thickness) {
            img[pixelIdx] = CIRCLE_COLOR;
        }

        pixelIdx += total_thread;
    }
}

__global__ void draw_bbox_kernel(uchar3* img,
                                 float3* data,   // bbox的四个点
                                 float2* grid, uchar* seam_mask, int thickness,
                                 int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        float3 p = make_float3(x, y, 1.0f);

#pragma unroll
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;

            float3 a = data[i];
            float3 b = data[j];

            float3 l = cross(a, b);

            float dis = point2line_dis(l, p);

            if (dis > thickness)
                continue;

            float3 La = make_float3(-1.0f * l.y, l.x, l.y * a.x - l.x * a.y);
            float3 Lb = make_float3(-1.0f * l.y, l.x, l.y * b.x - l.x * b.y);

            bool inside = dot(La, p) * dot(Lb, p) < 0 ? true : false;
            if (inside && seam_mask[pixelIdx] != 0) {
                img[pixelIdx] = BBOX_COLOR;
            }
        }

        pixelIdx += total_thread;
    }
}

// ================== Host ==============================

/**
 * 计算网格
 * 该函数将高度和宽度指定的数组H中的数据转换为grid中的浮点数对。
 *
 * @param H 输入数组，包含需要转换的数据。
 * @param grid 输出网格，存储转换后的浮点数对。
 * @param height 输入数组H的高度。
 * @param width 输入数组H的宽度。
 */
void compute_grid(float* H, float2* grid, int height, int width) {
    // 使用CUDA内核计算仿射变换后的图像
    compute_grid_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        H, grid, height, width);

    // 确保所有CUDA操作都已完成，检测错误
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 使用网格采样方法从源图像(src)中采样并重构目标图像(dst)。
 *
 * @param grid 指向存储仿射变换后图像数据的float2数组的指针。
 * @param src 指向源图像数据的uchar数组的指针。
 * @param dst 指向目标图像数据的uchar数组的指针。
 * @param src_h 源图像的高度。
 * @param src_w 源图像的宽度。
 * @param height 目标图像的高度。
 * @param width 目标图像的宽度。
 *
 * 该函数首先调用grid_sample_kernel内核，该内核使用CUDA并行计算来加速图像的采样过程。
 * 然后，通过调用cudaDeviceSynchronize()确保内核执行完成，接着使用cudaGetLastError()检查是否发生错误。
 */
void grid_sample(float2* grid, uchar* src, uchar* dst, int src_h, int src_w,
                 int height, int width) {
    // 调用CUDA内核进行网格采样
    grid_sample_kernel<uchar, 1><<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        grid, src, dst, src_h, src_w, height, width);

    // 确保内核执行完成，并检查是否有CUDA错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

void grid_sample(float2* grid, uchar3* src, uchar3* dst, int src_h, int src_w,
                 int height, int width) {
    grid_sample_kernel<uchar, 3><<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        grid, (uchar*)src, (uchar*)dst, src_h, src_w, height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 对给定的圆柱形图像进行下采样，生成缩放后的图像。
 *
 * @param img 圆柱形图像GPU数据结构，包含图像和掩码数据。
 * @param scale 缩放比例。
 * @param scaled_img 指向缩放后图像数据的指针。
 */
void pyramid_downsample(CylinderImageGPU img, int scale, uchar3* scaled_img) {
    // 计算需要进行的下采样级别
    int levels = log2f(scale) + 1;

    int height = img.height, width = img.width;

    // 分配当前图像的GPU内存
    short3* current_img;
    checkCudaErrors(
        cudaMalloc((void**)&current_img, sizeof(short3) * height * width));

    // 将输入图像从uchar3格式转换为short3格式，并进行归一化
    converTo_kernel<3><<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        (uchar*)img.image, img.mask, (short*)current_img, 32767.0f / 255.0f,
        height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 逐级下采样
    for (int i = 1; i < levels; i++) {
        height /= 2;
        width /= 2;

        // 分配下采样图像的GPU内存
        short3* _down_scale_img;
        checkCudaErrors(cudaMalloc((void**)&_down_scale_img,
                                   sizeof(short3) * height * width));

        // 执行下采样操作
        PyrDown_kernel<3><<<NUM_BLOCK(height * width), NUM_THREAD>>>(
            (short*)current_img, (short*)_down_scale_img, height, width);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        // 释放当前图像的GPU内存
        checkCudaErrors(cudaFree(current_img));

        // 更新当前图像指针为下采样后的图像
        current_img = _down_scale_img;
    }

    // 将最后一级下采样的图像转换回uchar3格式
    convertBack_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        current_img, scaled_img, height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 释放最后一级下采样图像的GPU内存
    checkCudaErrors(cudaFree(current_img));
}

/**
 * 计算两个图像的差异。
 *
 * @param img1
 * 指向第一个图像数据的指针，图像数据类型为uchar3（无符号字符3元素组）。
 * @param img2 指向第二个图像数据的指针，图像数据类型同样为uchar3。
 * @param output_diff 指向输出差异图像数据的指针，差异图像数据类型为ushort*。
 * @param temp_diff 用于中间计算的临时差异图像指针，类型为ushort*。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 该函数首先启动一个CUDA内核`diff_kernel`来计算两个输入图像的差异，
 * 然后使用CUDA API检查错误并确保计算完成。
 */
void compute_diff(uchar3* img1, uchar3* img2, ushort*& output_diff,
                  ushort*& temp_diff, int height, int width) {
    // 启动diff_kernel内核，使用NUM_BLOCK和NUM_THREAD配置块和线程
    diff_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        img1, img2, output_diff, height, width);

    // 确保所有CUDA作业已提交并执行完毕，检查有无CUDA错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 在指定的人区域更新差异值
 *
 * 该函数使用CUDA并行计算的方式，对给定图像中一个人物区域的差异值进行更新。它首先启动一个CUDA内核
 * `diff_person_kernel`
 * 来执行具体的计算任务，然后检查CUDA执行过程中是否有错误发生。
 *
 * @param diff 指向差异值数组的指针，该数组在GPU上进行更新。
 * @param grid 指向网格点坐标的浮点型二维数组，用于描述图像网格。
 * @param scale 缩放因子，用于调整处理的细节级别。
 * @param bbox 描述人物区域的边界框，包含x、y坐标以及宽度和高度。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 */
void update_diff_in_person_area(ushort* diff, float2* grid, int scale,
                                float4 bbox, int height, int width) {
    // 启动CUDA内核，对指定人物区域的差异值进行更新
    diff_person_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        diff, grid, scale, bbox.x, bbox.y, bbox.z, bbox.w, height, width);

    // 确保所有CUDA任务都已完成，检查是否有错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

void draw_circle(uchar3* img, float2 center, float radius, int thickness,
                 int height, int width) {
    draw_circle_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        img, center, radius, thickness, height, width);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

void draw_bbox(uchar3* img, float3* data, float2* grid, uchar* seam_mask,
               int thickness, int height, int width) {
    draw_bbox_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        img, data, grid, seam_mask, thickness, height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 在图像上绘制图标。
 *
 * @param img
 * 指向图像数据的指针，图像数据格式为uchar3，表示每个像素的红、绿、蓝三个颜色分量。
 * @param icon
 * 指向图标数据的指针，图标数据格式为uchar4，增加了透明通道alpha，用于控制图标透明度。
 * @param icon_h 图标的高度。
 * @param icon_w 图标的宽度。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 此函数首先调用draw_car_kernel内核函数，在图像上绘制图标。然后使用CUDA错误检查函数，
 * 确保内核调用和设备同步没有发生错误。
 */
void draw_icon(uchar3* img, uchar4* icon, int icon_h, int icon_w, int height,
               int width) {
    // 调用draw_car_kernel内核函数，使用NUM_BLOCK和NUM_THREAD指定的块和线程数
    draw_car_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        img, icon, icon_h, icon_w, height, width);

    // 确保所有CUDA操作都已完成，检查是否有错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 使用边缘缝合更新差异值
 *
 * 本函数通过上一次的缝合地图来更新输入输出的差异值数组。首先将缝合地图拷贝到设备上，然后调用
 * 核函数对差异值进行更新。最后释放缝合地图在设备上的内存，并确保所有CUDA操作完成。
 *
 * @param input_output_diff
 * 输入输出的差异值数组，类型为ushort*，在函数中被更新。
 * @param prev_seam_map 上一次的缝合地图，类型为cv::Mat，需为CV_8UC1类型。
 * @param height 输入输出差异值数组以及缝合地图的高度。
 * @param width 输入输出差异值数组以及缝合地图的宽度。
 */
void update_diff_use_seam(ushort* input_output_diff, cv::Mat prev_seam_map,
                          int height, int width) {
    uchar* prev_seam;
    // 分配设备内存以存放缝合地图
    checkCudaErrors(
        cudaMalloc((void**)&prev_seam, sizeof(uchar) * height * width));
    // 将缝合地图从主机内存拷贝到设备内存
    checkCudaErrors(cudaMemcpy(prev_seam, prev_seam_map.ptr<uchar>(0),
                               sizeof(uchar) * height * width,
                               cudaMemcpyHostToDevice));

    // 调用核函数，在设备上更新差异值
    update_diff_use_seam_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        input_output_diff, prev_seam, 1000, height, width);

    // 释放设备内存
    checkCudaErrors(cudaFree(prev_seam));
    // 确保所有CUDA操作已同步完成
    checkCudaErrors(cudaDeviceSynchronize());
    // 检查是否有CUDA错误发生
    checkCudaErrors(cudaGetLastError());
}
