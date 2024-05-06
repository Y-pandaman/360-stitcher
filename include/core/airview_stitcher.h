#ifndef __AIRVIEW_STITCHER__
#define __AIRVIEW_STITCHER__

#include "core/airview_stitcher_utils.h"
#include "stage/blend_kernel.h"
#include "util/helper_cuda.h"
#include "util/innoreal_timer.h"
#include "util/loguru.hpp"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

#define MIN_VALUE(a, b) a < b ? a : b
#define NUM_THREAD 512
#define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)

/**
 * BBox结构体定义了边界框的数据结构，包括视图索引和边界框的四个角点信息。
 */
struct BBox {
    int view_idx;       // 属于哪个视图的索引
    float4 data;        // 四个角点的浮点数表示
    float3* data_ptr;   // 指向四个线段的指针

    /**
     * BBox的默认构造函数。
     */
    BBox() { }

    /**
     * 构造函数，根据给定的信息和视图索引初始化边界框。
     * @param info 包含边界框中心点坐标和尺寸信息的float4类型。
     * @param view_idx_ 视图的索引。
     * @param height 图像的高度。
     * @param width 图像的宽度。
     */
    BBox(float4 info, int view_idx_, int height, int width) {
        // 根据信息计算并设置边界框的四个角点
        float x_center = info.x * width;
        float y_center = info.y * height;
        float x_min    = x_center - info.z * width / 2.0f;
        float y_min    = y_center - info.w * height / 2.0f;
        float x_max    = x_min + info.z * width;
        float y_max    = y_min + info.w * height;
        data           = make_float4(x_min, y_min, x_max, y_max);
        view_idx       = view_idx_;
    }

    /**
     * 构造函数，根据cv::Rect初始化边界框。
     * @param input 表示矩形区域的cv::Rect_<int>类型。
     * @param view_idx_ 视图的索引。
     */
    BBox(const cv::Rect_<int>& input, int view_idx_) {
        // 根据矩形区域设置边界框的四个角点
        float x_min = input.x;
        float y_min = input.y;
        float x_max = input.x + input.width;
        float y_max = input.y + input.height;
        data        = make_float4(x_min, y_min, x_max, y_max);
        view_idx    = view_idx_;
    }

    /**
     * 根据给定的单应性矩阵，获取映射后的四个角点。
     * @param H 单应性矩阵，是一个3x3的浮点数矩阵。
     * @param output 用于存储映射后四个角点的数组，数组元素类型为float3。
     * @return
     * 成功返回true，表示映射计算完成且结果有效，否则返回false，表示映射过程中发生错误或无法计算。
     */
    bool get_remapped_points(const cv::Mat& H, float3 output[4]) {
        // 计算原图边界框转换后的四个角点，并进行warp变换
        // 初始化边界框角点坐标
        float x_min = data.x, y_min = data.y, x_max = data.z, y_max = data.w;

        // 计算单应性矩阵应用后的x方向偏移
        float temp1 =
            -1.0f * H.at<float>(2, 0) * x_min - 1.0f * H.at<float>(2, 2);
        float temp2 =
            -1.0f * H.at<float>(2, 0) * x_max - 1.0f * H.at<float>(2, 2);

        // 根据单应性矩阵调整y_min的值，以确保映射后的点位于图像内
        if (H.at<float>(2, 1) > 0) {
            y_min = min(temp1 / H.at<float>(2, 1), temp2 / H.at<float>(2, 1)) -
                    1.0f;
        } else if (H.at<float>(2, 1) < 0) {
            y_min = max(temp1 / H.at<float>(2, 1), temp2 / H.at<float>(2, 1)) +
                    1.0f;
        } else {
            // 当单应性矩阵的某个系数为0时，检查是否会导致映射点位于图像边界外
            if (temp1 <= 0 || temp2 <= 0)
                return false;
        }

        // 执行warp变换并存储结果
        // 准备原始四个角点坐标
        float3 p[4] = {
            make_float3(x_min, y_min, 1), make_float3(x_max, y_min, 1),
            make_float3(x_max, y_max, 1), make_float3(x_min, y_max, 1)};

        // 对每个角点应用单应性矩阵并规范化坐标
        for (int i = 0; i < 4; i++) {
            float3 warped_p = homographBased_warp(p[i], H);   // 应用单应性变换

            assert(warped_p.z < 0);   // 确保变换后的点位于图像前方

            // 规范化坐标，去除齐次坐标中的z分量
            warped_p.x /= warped_p.z;
            warped_p.y /= warped_p.z;
            warped_p.z = 1;

            output[i] = warped_p;   // 存储变换后的角点坐标
        }
        return true;
    }

    /**
     * 根据单应性矩阵，将边界框进行映射变换并存储到data_ptr指向的设备内存中。
     * @param H 单应性矩阵。
     * @return 成功返回true，否则返回false。
     */
    bool remap(cv::Mat H) {
        // 计算原图边界框转换后的四个角点，并进行warp变换，结果存储到设备内存中
        float x_min = data.x, y_min = data.y, x_max = data.z, y_max = data.w;

        float temp1 =
            -1.0f * H.at<float>(2, 0) * x_min - 1.0f * H.at<float>(2, 2);
        float temp2 =
            -1.0f * H.at<float>(2, 0) * x_max - 1.0f * H.at<float>(2, 2);

        // 根据单应性矩阵调整y_min的值
        if (H.at<float>(2, 1) > 0) {
            y_min = min(temp1 / H.at<float>(2, 1), temp2 / H.at<float>(2, 1)) -
                    1.0f;
        } else if (H.at<float>(2, 1) < 0) {
            y_min = max(temp1 / H.at<float>(2, 1), temp2 / H.at<float>(2, 1)) +
                    1.0f;
        } else {
            if (temp1 <= 0 || temp2 <= 0)
                return false;
        }

        float3 p[4] = {
            make_float3(x_min, y_min, 1), make_float3(x_max, y_min, 1),
            make_float3(x_max, y_max, 1), make_float3(x_min, y_max, 1)};

        float3 warped_ps[4];
        for (int i = 0; i < 4; i++) {
            float3 warped_p = homographBased_warp(p[i], H);

            assert(warped_p.z < 0);

            warped_p.x /= warped_p.z;
            warped_p.y /= warped_p.z;
            warped_p.z = 1;

            warped_ps[i] = warped_p;
        }
        // 分配设备内存并复制warp变换后的结果
        cudaMalloc((void**)&data_ptr, sizeof(float3) * 4);
        cudaMemcpy(data_ptr, warped_ps, sizeof(float3) * 4,
                   cudaMemcpyHostToDevice);

        return true;
    }

    /**
     * 释放由remap方法分配的设备内存。
     */
    void free() {
        cudaFree(data_ptr);
    }
};

class AirViewStitcher {
    std::vector<float*> Hs_;   // Backwarp Homograph
    std::vector<float2*> grids_;
    std::vector<CylinderImageGPU> inputs_;
    std::vector<CylinderImageGPU> warped_inputs_;
    std::vector<uchar*> seam_masks_;
    std::vector<uchar3*> scaled_rgbs_;
    std::vector<ushort*> diffs_, temp_diffs_;
    uchar4* icon_rgba_;

    std::vector<cv::Mat> Hs_forward_;
    std::vector<cv::Mat> scaled_warped_masks_;
    std::vector<cv::Mat> overlap_masks_;
    cv::Mat total_mask_;
    // 使用bound_mask_将seam约束在上一帧的seam附近
    cv::Mat bound_mask_;
    int bound_kernel_ = 20;

    std::vector<int4> endPts_;
    std::vector<cv::Mat> diffs_map_;
    cv::Mat icon_;
    bool show_icon_ = false;

    uint64_t num_view_ = 0;
    int src_height_, src_width_, tgt_height_, tgt_width_;
    int size_src_, size_tgt_, size_scale_, size_icon_;
    int down_scale_seam_ = 1;
    int scale_height_, scale_width_;
    int icon_height_, icon_width_;

public:
    AirViewStitcher(int num_view, int src_height, int src_width, int tgt_height,
                    int tgt_width, int down_scale_seam);
    ~AirViewStitcher();
    void setIcon(std::string path, int new_width, int new_height);
    void init(std::vector<cv::Mat> input_mask, std::vector<cv::Mat> input_H);
    void feed(std::vector<cv::Mat> input_img,
              std::vector<std::vector<BBox>> bboxs);
    uchar3* output_GPUptr();
    cv::Mat output_CPUMat();
};

void compute_grid(float* H, float2* grid, int height, int width);

void grid_sample(float2* grid, uchar* src, uchar* dst, int src_h, int src_w,
                 int height, int width);
void grid_sample(float2* grid, uchar3* src, uchar3* dst, int src_h, int src_w,
                 int height, int width);

void pyramid_downsample(CylinderImageGPU img, int scale, uchar3* scaled_img);

void compute_diff(uchar3* img1, uchar3* img2, ushort*& output_diff,
                  ushort*& temp_diff, int height, int width);

void update_diff_use_seam(ushort* input_output_diff, cv::Mat prev_seam_map,
                          int height, int width);

void update_diff_in_person_area(ushort* diff, float2* grid, int scale,
                                float4 bbox, int height, int width);

void draw_circle(uchar3* img, float2 center, float radius, int thickness,
                 int height, int width);
void draw_bbox(uchar3* img, float3* data, float2* grid, uchar* seam_mask,
               int thickness, int height, int width);

void draw_icon(uchar3* img, uchar4* icon, int icon_h, int icon_w, int height,
               int width);

#endif
