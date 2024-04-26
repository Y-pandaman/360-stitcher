#include "project_to_cylinder.cuh"
#include <thrust/extrema.h>

static inline __device__ __host__ uchar3 interpolate1D(uchar3 v1, uchar3 v2,
                                                       float x) {
    return make_uchar3((uchar)((float)v1.x * (1.0f - x) + (float)v2.x * x),
                       (uchar)((float)v1.y * (1.0f - x) + (float)v2.y * x),
                       (uchar)((float)v1.z * (1.0f - x) + (float)v2.z * x));
}

static inline __device__ __host__ uchar3 interpolate2D(uchar3 v1, uchar3 v2,
                                                       uchar3 v3, uchar3 v4,
                                                       float x, float y) {
    uchar3 s = interpolate1D(v1, v2, x);
    uchar3 t = interpolate1D(v3, v4, x);
    return interpolate1D(s, t, y);
}

static inline __device__ uchar3 Bilinear(uchar3* src, int h, int w,
                                         float2 pixel) {
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

    // Condition Boundary  [TODO]

    return interpolate2D(src[idx00], src[idx10], src[idx01], src[idx11], x, y);
}

static __global__ void BackProjToSrc_kernel(
    uchar3* src_color, uchar* mask, int src_h, int src_w,
    CylinderGPU_stilib cyl, PinholeCameraGPU_stilib cam, uchar3* cyl_image,
    uchar* cyl_mask, int* uv, float* global_min_theta, float* global_min_phi,
    float* global_max_theta, float* global_max_phi, float* min_theta,
    float* min_phi, int height, int width, bool is_fisheye) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    float step_x = ((*global_max_theta) - (*global_min_theta)) / width;
    float step_y = ((*global_max_phi) - (*global_min_phi)) / height;

    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        float theta = x * step_x + *min_theta;
        float phi   = y * step_y + *min_phi;

        float3 P =
            make_float3(cyl.r * sinf(theta), cyl.r * phi, cyl.r * cosf(theta));

        float2 pixel;
        if (is_fisheye) {
            pixel = cam.projWorldToPixelFishEye(cyl.rotateVector_inv(P) +
                                                cyl.getCenter());
        } else {
            pixel =
                cam.projWorldToPixel(cyl.rotateVector_inv(P) + cyl.getCenter());
        }

        uchar3 color = make_uchar3(0, 0, 0);
        uchar m      = 0;

        if (mask == nullptr) {
            if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < src_w - 1 &&
                pixel.y < src_h - 1) {
                color = Bilinear(src_color, src_h, src_w, pixel);
            }

            int row = (phi - (*global_min_phi)) / step_y + 0.5;
            int col = (theta - (*global_min_theta)) / step_x + 0.5;
            if (x == 0 && y == 0) {
                uv[0] = 0;
                uv[1] = 0;
            }

            if (row >= 0 && row < height && col >= 0 && col < width) {
                cyl_image[row * width + col] = color;
            }
        } else {
            if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < src_w - 1 &&
                pixel.y < src_h - 1) {
                color = Bilinear(src_color, src_h, src_w, pixel);
                m = mask[(int)(pixel.y + 0.5f) * src_w + (int)(pixel.x + 0.5f)];
            }

            int row = (phi - (*global_min_phi)) / step_y + 0.5;
            int col = (theta - (*global_min_theta)) / step_x + 0.5;
            if (x == 0 && y == 0) {
                uv[0] = 0;
                uv[1] = 0;
            }

            if (row >= 0 && row < height && col >= 0 && col < width) {
                cyl_image[row * width + col] = color;
                cyl_mask[row * width + col]  = m;
            }
        }

        pixelIdx += total_thread;
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

