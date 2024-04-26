#include "airview_stitcher.h"

__device__ __constant__ uchar3 BBOX_COLOR   = {50, 205, 50};
__device__ __constant__ uchar3 CIRCLE_COLOR = {255, 144, 30};

// ================= DEVICE ==================================

template <typename T, int CHANNEL>
static inline __device__ void Bilinear(T* src, T* dst, int h, int w,
                                       float2 pixel) {
    // Condition Boundary  [TODO]
    if (pixel.x < 1 || pixel.x >= w - 1 || pixel.y < 1 || pixel.y >= h - 1) {
#pragma unroll
        for (int i = 0; i < CHANNEL; i++) {
            dst[i] = 0;
        }
        return;
    }
    int x0  = (int)pixel.x;
    int y0  = (int)pixel.y;
    int x1  = x0 + 1;
    int y1  = y0 + 1;
    float x = pixel.x - x0;
    float y = pixel.y - y0;

    int idx00 = x0 + y0 * w;
    int idx01 = idx00 + w;
    int idx10 = idx00 + 1;
    int idx11 = idx01 + 1;

#pragma unroll
    for (int i = 0; i < CHANNEL; i++) {
        dst[i] = (T)((float)src[idx00 * CHANNEL + i] * (1.0f - x) * (1.0f - y) +
                     (float)src[idx01 * CHANNEL + i] * (1.0f - x) * y +
                     (float)src[idx10 * CHANNEL + i] * x * (1.0f - y) +
                     (float)src[idx11 * CHANNEL + i] * x * y);
    }
}

template <typename T, int CHANNEL>
__device__ static inline void weight_product(float* color, T* src, float& aw,
                                             float w) {
#pragma unroll
    for (int i = 0; i < CHANNEL; i++) {
        color[i] = color[i] + (float)src[i] * w;
    }
    aw += w;
}

__device__ static inline float point2line_dis(float3 L, float3 P) {
    return fabs(dot(L, P)) / sqrt(L.x * L.x + L.y * L.y);
}

// ================= Kernel ===================================
__global__ void compute_grid_kernel(float* H, float2* grid, int height,
                                    int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        float x_ = H[0] * x + H[1] * y + H[2];
        float y_ = H[3] * x + H[4] * y + H[5];
        float z_ = H[6] * x + H[7] * y + H[8];

        if (z_ < 0) {
            float u = x_ / z_;
            float v = y_ / z_;

            grid[pixelIdx] = make_float2(u, v);

        } else {
            grid[pixelIdx] = make_float2(-1.0f, -1.0f);
        }
        pixelIdx += total_thread;
    }
}

template <typename T, int CHANNEL>
__global__ void grid_sample_kernel(float2* grid, T* src, T* dst, int src_h,
                                   int src_w, int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    while (pixelIdx < totalPixel) {
        Bilinear<T, CHANNEL>(src, dst + pixelIdx * CHANNEL, src_h, src_w,
                             grid[pixelIdx]);

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
template <int CHANNEL>
__global__ void PyrDown_kernel(short* src, short* dst, int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        int src_x        = 2 * (pixelIdx % width);
        int src_y        = 2 * (pixelIdx / width);
        int src_pixelIdx = src_y * (width * 2) + src_x;

        float color[CHANNEL];
        for (int i = 0; i < CHANNEL; i++)
            color[i] = 0;
        float weight = 0.0f;

#pragma unroll
        for (int i = 0; i < 5; i++) {
#pragma unroll
            for (int j = 0; j < 5; j++) {
                int u = src_x + j - 2;
                int v = src_y + i - 2;

                if (u < 0 || v < 0 || u >= width * 2 || v >= height * 2)
                    continue;

                if (src[CHANNEL * (v * 2 * width + u)] == -32768)
                    continue;

                weight_product<short, CHANNEL>(
                    color, src + CHANNEL * (v * 2 * width + u), weight,
                    GauKernel[i * 5 + j]);
            }
        }
#pragma unroll

        if (weight == 0) {
            for (int i = 0; i < CHANNEL; i++) {
                dst[CHANNEL * pixelIdx + i] = -32768;
            }
        } else {
            for (int i = 0; i < CHANNEL; i++) {
                color[i]                    = color[i] / weight;
                dst[CHANNEL * pixelIdx + i] = (short)color[i];
            }
        }

        pixelIdx += total_thread;
    }
}

__global__ void convertBack_kernel(short3* src, uchar3* dst, int height,
                                   int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        dst[pixelIdx].x = (uchar)((float)src[pixelIdx].x / 32767.0f * 255.0f);
        dst[pixelIdx].y = (uchar)((float)src[pixelIdx].y / 32767.0f * 255.0f);
        dst[pixelIdx].z = (uchar)((float)src[pixelIdx].z / 32767.0f * 255.0f);

        pixelIdx += total_thread;
    }
}

template <int CHANNEL>
__global__ void converTo_kernel(uchar* src, uchar* mask, short* dst,
                                float scale, int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
#pragma unroll
        for (int i = 0; i < CHANNEL; i++) {
            dst[CHANNEL * pixelIdx + i] =
                (float)src[CHANNEL * pixelIdx + i] * scale;
        }

        if (CHANNEL == 3 && mask[pixelIdx] == 0) {
#pragma unroll
            for (int i = 0; i < CHANNEL; i++) {
                dst[CHANNEL * pixelIdx + i] = -32768;
            }
        }

        pixelIdx += total_thread;
    }
}

__global__ void diff_kernel(uchar3* img1, uchar3* img2, ushort* diff,
                            int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        uchar3 rgb1 = img1[pixelIdx];
        uchar3 rgb2 = img2[pixelIdx];

        int sum1 = rgb1.x + rgb1.y + rgb1.z;
        int sum2 = rgb2.x + rgb2.y + rgb2.z;

        if (sum1 == 0 || sum2 == 0) {
            diff[pixelIdx] = 65535;
            pixelIdx += total_thread;
            continue;
        }

        float3 c1 =
            make_float3((float)rgb1.x, (float)rgb1.y, (float)rgb1.z) / MASK_MAX;
        float3 c2 =
            make_float3((float)rgb2.x, (float)rgb2.y, (float)rgb2.z) / MASK_MAX;
        float3 d = fabs(c1 - c2);

        float base = 0.5;
        diff[pixelIdx] =
            (ushort)((d.x + d.y + d.z + base) / (3.0f + base) * 65535);

        pixelIdx += total_thread;
    }
}

__global__ void update_diff_use_seam_kernel(ushort* input_output_diff,
                                            uchar* seam, ushort delta_on_seam,
                                            int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        if (seam[pixelIdx] != 255 && input_output_diff[pixelIdx] != 65535)
            input_output_diff[pixelIdx] = 0;
        pixelIdx += total_thread;
    }
}

__global__ void diff_person_kernel(ushort* diff, float2* grid, int scale,
                                   float x_min, float y_min, float x_max,
                                   float y_max, int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        float2 uv = grid[y * scale * (width * scale) + x * scale];
        if (uv.x >= x_min && uv.x <= x_max && uv.y >= y_min && uv.y <= y_max) {
            diff[pixelIdx] = 65535;
        }

        pixelIdx += total_thread;
    }
}

__global__ void draw_car_kernel(uchar3* img, uchar4* icon, int icon_h,
                                int icon_w, int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    int cx     = width / 2;
    int cy     = height / 2;
    int half_w = icon_w / 2;
    int half_h = icon_h / 2;

    int min_x = cx - half_w;
    int min_y = cy - half_h;
    int max_x = min_x + icon_w;
    int max_y = min_y + icon_h;

    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        if (x >= min_x && x < max_x && y >= min_y && y < max_y) {
            uchar4 rgba = icon[(y - min_y) * icon_w + (x - min_x)];

            if (rgba.w > 128) {
                img[pixelIdx] = make_uchar3(rgba.x, rgba.y, rgba.z);
            }
        }

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

void compute_grid(float* H, float2* grid, int height, int width) {
    compute_grid_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        H, grid, height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

void grid_sample(float2* grid, uchar* src, uchar* dst, int src_h, int src_w,
                 int height, int width) {
    grid_sample_kernel<uchar, 1><<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        grid, src, dst, src_h, src_w, height, width);
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

void pyramid_downsample(CylinderImageGPU img, int scale, uchar3* scaled_img) {
    int levels = log2f(scale) + 1;

    int height = img.height, width = img.width;

    short3* current_img;
    checkCudaErrors(
        cudaMalloc((void**)&current_img, sizeof(short3) * height * width));

    converTo_kernel<3><<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        (uchar*)img.image, img.mask, (short*)current_img, 32767.0f / 255.0f,
        height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    for (int i = 1; i < levels; i++) {
        height /= 2;
        width /= 2;

        short3* _down_scale_img;
        checkCudaErrors(cudaMalloc((void**)&_down_scale_img,
                                   sizeof(short3) * height * width));

        PyrDown_kernel<3><<<NUM_BLOCK(height * width), NUM_THREAD>>>(
            (short*)current_img, (short*)_down_scale_img, height, width);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaFree(current_img));

        current_img = _down_scale_img;
    }

    convertBack_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        current_img, scaled_img, height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(current_img));
}

void compute_diff(uchar3* img1, uchar3* img2, ushort*& output_diff,
                  ushort*& temp_diff, int height, int width) {
    diff_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        img1, img2, output_diff, height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

void update_diff_in_person_area(ushort* diff, float2* grid, int scale,
                                float4 bbox, int height, int width) {
    diff_person_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        diff, grid, scale, bbox.x, bbox.y, bbox.z, bbox.w, height, width);

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

void draw_icon(uchar3* img, uchar4* icon, int icon_h, int icon_w, int height,
               int width) {
    draw_car_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        img, icon, icon_h, icon_w, height, width);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

void update_diff_use_seam(ushort* input_output_diff, cv::Mat prev_seam_map,
                          int height, int width) {
    uchar* prev_seam;
    checkCudaErrors(
        cudaMalloc((void**)&prev_seam, sizeof(uchar) * height * width));
    checkCudaErrors(cudaMemcpy(prev_seam, prev_seam_map.ptr<uchar>(0),
                               sizeof(uchar) * height * width,
                               cudaMemcpyHostToDevice));

    update_diff_use_seam_kernel<<<NUM_BLOCK(height * width), NUM_THREAD>>>(
        input_output_diff, prev_seam, 1000, height, width);

    checkCudaErrors(cudaFree(prev_seam));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}
