#include "project_to_cylinder.cuh"
#include <thrust/extrema.h>

// 在两个像素间进行线性插值
static inline __device__ __host__ uchar3 interpolate1D(uchar3 v1, uchar3 v2,
                                                       float x) {
    return make_uchar3((uchar)((float)v1.x * (1.0f - x) + (float)v2.x * x),
                       (uchar)((float)v1.y * (1.0f - x) + (float)v2.y * x),
                       (uchar)((float)v1.z * (1.0f - x) + (float)v2.z * x));
}

// 使用interpolate1D来首先在一个方向上进行插值，然后在另一个方向上进行插值，从而实现双线性插值
static inline __device__ __host__ uchar3 interpolate2D(uchar3 v1, uchar3 v2,
                                                       uchar3 v3, uchar3 v4,
                                                       float x, float y) {
    uchar3 s = interpolate1D(v1, v2, x);
    uchar3 t = interpolate1D(v3, v4, x);
    return interpolate1D(s, t, y);
}

// 双线性插值算法
// 根据给定的浮点坐标pixel在源图像src中找到最接近的四个像素，并使用双线性插值来计算该位置的颜色值。
static inline __device__ uchar3 Bilinear(uchar3* src, int h, int w,
                                         float2 pixel) {
    // 计算四个角像素的整数索引
    int x0 = max(0, (int)pixel.x);   // 使用max确保索引不小于0
    int y0 = max(0, (int)pixel.y);   // 使用max确保索引不小于0
    int x1 = min(w - 1, x0 + 1);   // 使用min确保索引不超过图像宽度
    int y1 = min(h - 1, y0 + 1);   // 使用min确保索引不超过图像高度

    // 计算插值的相对位置
    float x = pixel.x - x0;
    float y = pixel.y - y0;

    // 计算四个角像素的一维索引
    int idx00 = x0 + y0 * w;
    int idx01 = idx00 + w;
    int idx10 = idx00 + 1;
    int idx11 = idx01 + 1;

    // 使用clamp确保插值权重在0到1之间
    float x_clamped = clamp(x, 0.0f, 1.0f);
    float y_clamped = clamp(y, 0.0f, 1.0f);

    // 对四个角像素进行双线性插值
    return interpolate2D(src[idx00], src[idx10], src[idx01], src[idx11],
                         x_clamped, y_clamped);
}

// 将圆柱体图像投影回原始视角
static __global__ void BackProjToSrc_kernel(
    uchar3* src_color, uchar* mask, int src_h, int src_w,
    CylinderGPU_stilib cyl, PinholeCameraGPU_stilib cam, uchar3* cyl_image,
    uchar* cyl_mask, int* uv, float* global_min_theta, float* global_min_phi,
    float* global_max_theta, float* global_max_phi, float* min_theta,
    float* min_phi, int height, int width, bool is_fisheye) {
    // 线程和像素索引计算
    // pixelIdx计算当前线程处理的像素索引，total_thread计算总的线程数
    // totalPixel计算需要处理的总像素数。
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 计算行和列的步长
    float step_x = ((*global_max_theta) - (*global_min_theta)) / width;
    float step_y = ((*global_max_phi) - (*global_min_phi)) / height;

    // 循环每个像素
    while (pixelIdx < totalPixel) {
        // 计算当前像素的坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 计算圆柱体坐标
        float theta = x * step_x + *min_theta;
        float phi   = y * step_y + *min_phi;

        // 使用 theta 和 phi 计算世界坐标 P
        float3 P =
            make_float3(cyl.r * sinf(theta), cyl.r * phi, cyl.r * cosf(theta));

        // 将世界坐标 P 反投影到源图像空间的像素坐标 pixel
        float2 pixel;
        if (is_fisheye) {
            pixel = cam.projWorldToPixelFishEye(cyl.rotateVector_inv(P) +
                                                cyl.getCenter());
        } else {
            pixel =
                cam.projWorldToPixel(cyl.rotateVector_inv(P) + cyl.getCenter());
        }

        // 初始化颜色 color 为黑色，掩码 m 为0
        uchar3 color = make_uchar3(0, 0, 0);
        uchar m      = 0;

        if (mask == nullptr) {
            // 确保反投影的像素坐标在源图像的边界内，然后使用双线性插值 Bilinear
            // 计算颜色
            if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < src_w - 1 &&
                pixel.y < src_h - 1) {
                color = Bilinear(src_color, src_h, src_w, pixel);
            }
            int row = (phi - (*global_min_phi)) / step_y + 0.5;
            int col = (theta - (*global_min_theta)) / step_x + 0.5;
            // 更新纹理坐标
            // 如果当前处理的是第一个像素（x == 0 && y == 0），则更新 uv 数组。
            if (x == 0 && y == 0) {
                uv[0] = 0;
                uv[1] = 0;
            }

            // 更新 cyl_image
            if (row >= 0 && row < height && col >= 0 && col < width) {
                cyl_image[row * width + col] = color;
            }
        } else {
            // 确保反投影的像素坐标在源图像的边界内，然后使用双线性插值 Bilinear
            // 计算颜色和计算掩码值 m
            if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < src_w - 1 &&
                pixel.y < src_h - 1) {
                color = Bilinear(src_color, src_h, src_w, pixel);
                m = mask[(int)(pixel.y + 0.5f) * src_w + (int)(pixel.x + 0.5f)];
            }
            int row = (phi - (*global_min_phi)) / step_y + 0.5;
            int col = (theta - (*global_min_theta)) / step_x + 0.5;
            // 更新纹理坐标
            // 如果当前处理的是第一个像素（x == 0 && y == 0），则更新 uv 数组。
            if (x == 0 && y == 0) {
                uv[0] = 0;
                uv[1] = 0;
            }
            // 更新 cyl_image 和 cyl_mask
            if (row >= 0 && row < height && col >= 0 && col < width) {
                cyl_image[row * width + col] = color;
                cyl_mask[row * width + col]  = m;
            }
        }

        pixelIdx += total_thread;   // 在处理完一个像素后，通过增加 pixelIdx
                                    // 来移动到下一个像素。
    }
}

bool projToCylinderImage_cuda(ViewGPU_stilib view,
                              CylinderImageGPU_stilib& cyl_image,
                              CylinderGPU_stilib& cylinder, int cyl_image_width,
                              int cyl_image_height) {
    // Suppose all images are same size; NOTE: FIX IF NECESSARY
    int height = view.height, width = view.width;
    int size_c = cylinder.offset[3];

    int num_thread = 512;
    int num_block  = min(65535, (size_c + num_thread - 1) / num_thread);
    int num_block2 =
        min(65535,
            (cyl_image_width * cyl_image_height + num_thread - 1) / num_thread);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    BackProjToSrc_kernel<<<num_block2, num_thread>>>(
        view.image, view.mask, view.height, view.width, cylinder, view.camera,
        cyl_image.image, cyl_image.mask, cyl_image.uv, cylinder.global_theta,
        cylinder.global_phi, cylinder.global_theta + 1, cylinder.global_phi + 1,
        cylinder.global_theta, cylinder.global_phi, cyl_image_height,
        cyl_image_width, true);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    return true;
}
