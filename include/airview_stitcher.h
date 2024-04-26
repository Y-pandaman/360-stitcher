#ifndef __AIRVIEW_STITCHER__
#define __AIRVIEW_STITCHER__

#include "airview_stitcher_utils.h"
#include "blend.h"
#include "helper_cuda.h"
#include "innoreal_timer.hpp"
#include "loguru.hpp"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

#define MIN_VALUE(a, b) a < b ? a : b
#define NUM_THREAD 512
#define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)

//#define OUTPUT_WARPED_RGB
//#define OUTPUT_TOTAL_SEAM_MAP
//#define OUTPUT_DIFF_IMAGE
//#define OUTPUT_FINAL_DIFF

struct BBox {
    int view_idx;       // 属于哪个view
    float4 data;        // 4  corners
    float3* data_ptr;   // 4 lines

    BBox() { }
    BBox(float4 info, int view_idx_, int height, int width) {
        float x_center = info.x * width;
        float y_center = info.y * height;
        float x_min    = x_center - info.z * width / 2.0f;
        float y_min    = y_center - info.w * height / 2.0f;
        float x_max    = x_min + info.z * width;
        float y_max    = y_min + info.w * height;
        data           = make_float4(x_min, y_min, x_max, y_max);
        view_idx       = view_idx_;
    }

    BBox(const cv::Rect_<int>& input, int view_idx_) {
        float x_min = input.x;
        float y_min = input.y;
        float x_max = input.x + input.width;
        float y_max = input.y + input.height;
        data        = make_float4(x_min, y_min, x_max, y_max);
        view_idx    = view_idx_;
    }

    bool get_remapped_points(const cv::Mat& H, float3 output[4]) {
        // 现在原图转换成框， 然后warp
        float x_min = data.x, y_min = data.y, x_max = data.z, y_max = data.w;

        float3 test_point = make_float3(x_min, y_min, 1);

        float temp1 =
            -1.0f * H.at<float>(2, 0) * x_min - 1.0f * H.at<float>(2, 2);
        float temp2 =
            -1.0f * H.at<float>(2, 0) * x_max - 1.0f * H.at<float>(2, 2);

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

        for (int i = 0; i < 4; i++) {
            float3 warped_p = homographBased_warp(p[i], H);

            assert(warped_p.z < 0);

            warped_p.x /= warped_p.z;
            warped_p.y /= warped_p.z;
            warped_p.z = 1;

            output[i] = warped_p;
        }
        return true;
    }

    bool remap(cv::Mat H) {
        // 现在原图转换成框， 然后warp
        float x_min = data.x, y_min = data.y, x_max = data.z, y_max = data.w;

        float3 test_point = make_float3(x_min, y_min, 1);

        float temp1 =
            -1.0f * H.at<float>(2, 0) * x_min - 1.0f * H.at<float>(2, 2);
        float temp2 =
            -1.0f * H.at<float>(2, 0) * x_max - 1.0f * H.at<float>(2, 2);

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
        cudaMalloc((void**)&data_ptr, sizeof(float3) * 4);
        cudaMemcpy(data_ptr, warped_ps, sizeof(float3) * 4,
                   cudaMemcpyHostToDevice);

        return true;
    }

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

    int num_view_ = 0;
    int src_height_, src_width_, tgt_height_, tgt_width_;
    int size_src_, size_tgt_, size_scale_, size_icon_;
    int down_scale_seam_ = 1;
    int scale_height_, scale_width_;
    int icon_height_, icon_width_;

private:
    cv::Mat pano_seam_assist_comp;
    std::vector<cv::Mat> pano_seam_assist_vec;
    cv::Mat getSeamRangeMask(int id);

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
