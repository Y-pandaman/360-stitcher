#include "airview_stitcher.h"

AirViewStitcher::AirViewStitcher(int num_view, int src_height, int src_width,
                                 int tgt_height, int tgt_width,
                                 int down_scale_seam) {
    num_view_   = num_view;
    src_height_ = src_height;
    src_width_  = src_width;
    tgt_height_ = tgt_height;
    tgt_width_  = tgt_width;

    size_src_ = src_height_ * src_width_;
    size_tgt_ = tgt_height_ * tgt_width_;

    down_scale_seam_ = down_scale_seam;

    scale_height_ = tgt_height_ / down_scale_seam_;
    scale_width_  = tgt_width_ / down_scale_seam_;
    size_scale_   = scale_height_ * scale_width_;

    while (num_view > 0) {
        inputs_.emplace_back(src_height, src_width);
        warped_inputs_.emplace_back(tgt_height, tgt_width);

        uchar* seam_mask;

        checkCudaErrors(
            cudaMalloc((void**)&seam_mask, sizeof(uchar) * size_tgt_));
        seam_masks_.emplace_back(seam_mask);
        float* H;
        checkCudaErrors(cudaMalloc((void**)&H, sizeof(float) * 9));
        Hs_.emplace_back(H);

        float2* grid;
        checkCudaErrors(cudaMalloc((void**)&grid, sizeof(float2) * size_tgt_));
        grids_.emplace_back(grid);

        uchar3* scaled_rgb;
        checkCudaErrors(
            cudaMalloc((void**)&scaled_rgb, sizeof(uchar3) * size_scale_));
        scaled_rgbs_.emplace_back(scaled_rgb);

        ushort* diff;
        checkCudaErrors(
            cudaMalloc((void**)&diff, sizeof(ushort) * size_scale_));
        diffs_.emplace_back(diff);
        checkCudaErrors(
            cudaMalloc((void**)&diff, sizeof(ushort) * size_scale_));
        temp_diffs_.emplace_back(diff);

        num_view--;
    }
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("[ERROR] Create AirViewStitcher failed !!! \n");
    } else {
        printf("Create AirViewStitcher successfully!  num_view: %d\n",
               num_view_);
    }
}

AirViewStitcher::~AirViewStitcher() {
    printf("[TODO] Not finished yet\n");
    printf("Destroyed AirViewStitcher\n");
}

void AirViewStitcher::setIcon(std::string path, int new_width, int new_height) {
    icon_      = cv::imread(path, cv::IMREAD_UNCHANGED);
    show_icon_ = true;
    std::cout << "load icon " << icon_.rows << " " << icon_.cols << " "
              << icon_.channels() << std::endl;

    icon_height_ = new_height;
    icon_width_  = new_width;

    size_icon_ = icon_height_ * icon_width_;

    cv::resize(icon_, icon_, cv::Size(icon_width_, icon_height_));

    cudaMalloc((void**)&icon_rgba_, sizeof(uchar4) * size_icon_);
    cudaMemcpy(icon_rgba_, icon_.ptr<uchar4>(0), sizeof(uchar4) * size_icon_,
               cudaMemcpyHostToDevice);
}

void AirViewStitcher::init(std::vector<cv::Mat> input_masks,
                           std::vector<cv::Mat> input_Hs) {
    assert(input_masks.size() == num_view_);
    assert(input_Hs.size() == num_view_);

    scaled_warped_masks_.resize(num_view_);

    for (int i = 0; i < num_view_; i++) {
        Hs_forward_.emplace_back(input_Hs[i]);
        checkCudaErrors(
            cudaMemcpy(inputs_[i].mask, input_masks[i].ptr<uchar>(0),
                       sizeof(uchar) * size_src_, cudaMemcpyHostToDevice));
        cv::Mat H = input_Hs[i].inv();
        checkCudaErrors(cudaMemcpy(Hs_[i], H.ptr<float>(0), sizeof(float) * 9,
                                   cudaMemcpyHostToDevice));

        compute_grid(Hs_[i], grids_[i], tgt_height_, tgt_width_);

        grid_sample(grids_[i], inputs_[i].mask, warped_inputs_[i].mask,
                    src_height_, src_width_, tgt_height_, tgt_width_);

        uchar* _mask;
        checkCudaErrors(cudaHostAlloc((void**)&_mask, sizeof(uchar) * size_tgt_,
                                      cudaHostAllocDefault));
        checkCudaErrors(cudaMemcpy(_mask, warped_inputs_[i].mask,
                                   sizeof(uchar) * size_tgt_,
                                   cudaMemcpyDeviceToHost));
        scaled_warped_masks_[i] =
            cv::Mat(tgt_height_, tgt_width_, CV_8UC1, _mask);

        cv::resize(scaled_warped_masks_[i], scaled_warped_masks_[i],
                   cv::Size(scale_width_, scale_height_));
        cv::erode(scaled_warped_masks_[i], scaled_warped_masks_[i],
                  getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));
        cv::threshold(scaled_warped_masks_[i], scaled_warped_masks_[i], 128,
                      255, cv::THRESH_BINARY);
        cv::imwrite("warped_mask" + std::to_string(i) + ".png",
                    scaled_warped_masks_[i]);
    }

    overlap_masks_.resize(num_view_);
    endPts_.resize(num_view_);
    diffs_map_.resize(num_view_);
    total_mask_ = cv::Mat::zeros(scale_height_, scale_width_, CV_8UC1);
    bound_mask_ = cv::Mat::ones(scale_height_, scale_width_, CV_8UC1) * 255;

    for (int i = 0; i < num_view_; i++) {
        int j = (i + 1) % num_view_;

        overlap_masks_[i] = scaled_warped_masks_[i] & scaled_warped_masks_[j];
        total_mask_       = total_mask_ | scaled_warped_masks_[i];
        diffs_map_[i] = cv::Mat::zeros(scale_height_, scale_width_, CV_16UC1);
    }

    for (int i = 0; i < num_view_; i++) {
        int j = (i + 1) % num_view_;
        endPts_[i] =
            decide_start_end(overlap_masks_[i], scaled_warped_masks_[i],
                             scaled_warped_masks_[j]);
        printf("i %d start (%d, %d)  end (%d, %d)\n", i, endPts_[i].x,
               endPts_[i].y, endPts_[i].z, endPts_[i].w);
    }

    cv::erode(total_mask_, total_mask_,
              getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("[ERROR] Init AirViewStitcher failed !!! \n");
    } else {
        printf("Init AirViewStitcher successfully!  num_view: %d\n", num_view_);
    }
}

void AirViewStitcher::feed(std::vector<cv::Mat> input_img,
                           std::vector<std::vector<BBox>> bboxs) {
    assert(input_img.size() == num_view_);
    assert(bboxs.size() == num_view_);

#ifdef OUTPUT_WARPED_RGB
    static int frame_warped_rgb_count = 0;
#endif

    for (int i = 0; i < num_view_; i++) {
        checkCudaErrors(
            cudaMemcpy(inputs_[i].image, input_img[i].ptr<uchar3>(0),
                       sizeof(uchar3) * size_src_, cudaMemcpyHostToDevice));

        grid_sample(grids_[i], inputs_[i].image, warped_inputs_[i].image,
                    src_height_, src_width_, tgt_height_, tgt_width_);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
#ifdef OUTPUT_WARPED_RGB
        cv::Mat rgb, mask;
        warped_inputs_[i].toCPU(rgb, mask);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        std::filesystem::path warped_image_path("./output/warped_image/");
        warped_image_path.append(std::to_string(i));
        if (!std::filesystem::exists(warped_image_path)) {
            std::filesystem::create_directories(warped_image_path);
        }
        warped_image_path.append("warped_rgb_" + std::to_string(i) + "-" +
                                 std::to_string(frame_warped_rgb_count) +
                                 ".png");
        cv::imwrite(warped_image_path, rgb);
#endif
        pyramid_downsample(warped_inputs_[i], down_scale_seam_,
                           scaled_rgbs_[i]);
    }

#ifdef OUTPUT_WARPED_RGB
    frame_warped_rgb_count++;
#endif

#ifdef OUTPUT_DIFF_IMAGE
    static int diff_frame_count = 0;
#endif

#ifdef OUTPUT_FINAL_DIFF
    static int diff_final_frame_count = 0;
#endif

    static cv::Mat prev_frame_seam;

    // compute diff
    for (int i = 0; i < num_view_; i++) {
        int j = (i + 1) % num_view_;
        compute_diff(scaled_rgbs_[i], scaled_rgbs_[j], diffs_[i],
                     temp_diffs_[i], scale_height_, scale_width_);
#ifdef OUTPUT_DIFF_IMAGE
        int height = scale_height_, width = scale_width_;
        ushort* output_img_data;
        checkCudaErrors(cudaHostAlloc((void**)&output_img_data,
                                      sizeof(ushort) * height * width,
                                      cudaHostAllocDefault));
        checkCudaErrors(cudaMemcpy(output_img_data, diffs_[i],
                                   sizeof(ushort) * height * width,
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        cv::Mat img = cv::Mat(height, width, CV_16UC1, output_img_data);
        std::filesystem::path diff_img_dilated_path(
            "./output/diff_img_dilated/");
        diff_img_dilated_path.append("diff_img_dilated_" + std::to_string(i));
        if (!std::filesystem::exists(diff_img_dilated_path)) {
            std::filesystem::create_directories(diff_img_dilated_path);
        }

        diff_img_dilated_path.append("diff_img_dilated_" + std::to_string(i) +
                                     "-" + std::to_string(diff_frame_count) +
                                     ".png");
        cv::imwrite(diff_img_dilated_path, img);

        std::filesystem::path diff_img_path("./output/diff_img");
        diff_img_path.append("diff_img_" + std::to_string(i));
        if (!std::filesystem::exists(diff_img_path)) {
            std::filesystem::create_directories(diff_img_path);
        }
        diff_img_path.append("diff_img_" + std::to_string(i) + "-" +
                             std::to_string(diff_frame_count) + ".png");

        checkCudaErrors(cudaMemcpy(output_img_data, temp_diffs_[i],
                                   sizeof(ushort) * height * width,
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        img = cv::Mat(height, width, CV_16UC1, output_img_data);
        cv::imwrite(diff_img_path, img);

        checkCudaErrors(cudaFreeHost(output_img_data));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
#endif
        std::vector<BBox> boundingboxs_i = bboxs[i];
        std::vector<BBox> boundingboxs_j = bboxs[j];
        for (int k = 0; k < boundingboxs_i.size(); k++) {
            update_diff_in_person_area(diffs_[i], grids_[i], down_scale_seam_,
                                       boundingboxs_i[k].data, scale_height_,
                                       scale_width_);
        }
        for (int k = 0; k < boundingboxs_j.size(); k++) {
            update_diff_in_person_area(diffs_[i], grids_[j], down_scale_seam_,
                                       boundingboxs_j[k].data, scale_height_,
                                       scale_width_);
        }
        if (!prev_frame_seam.empty()) {
            update_diff_use_seam(diffs_[i], prev_frame_seam, scale_height_,
                                 scale_width_);
        }
        cudaMemcpy(diffs_map_[i].ptr<ushort>(0), diffs_[i],
                   sizeof(ushort) * size_scale_, cudaMemcpyDeviceToHost);
#ifdef OUTPUT_FINAL_DIFF
        ushort* final_diff_img;
        checkCudaErrors(
            cudaHostAlloc((void**)&final_diff_img,
                          sizeof(ushort) * scale_height_ * scale_width_,
                          cudaHostAllocDefault));
        checkCudaErrors(
            cudaMemcpy(final_diff_img, diffs_[i],
                       sizeof(ushort) * scale_height_ * scale_width_,
                       cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        cv::Mat final_diff_img_mat =
            cv::Mat(scale_height_, scale_width_, CV_16UC1, final_diff_img);
        std::filesystem::path diff_img_final_path("./output/diff_final_img/");
        diff_img_final_path.append("diff_final_img_" + std::to_string(i));
        if (!std::filesystem::exists(diff_img_final_path)) {
            std::filesystem::create_directories(diff_img_final_path);
        }
        diff_img_final_path.append("diff_img_final_" + std::to_string(i) + "-" +
                                   std::to_string(diff_final_frame_count) +
                                   ".png");
        cv::imwrite(diff_img_final_path, final_diff_img_mat);
#endif
    }

#ifdef OUTPUT_DIFF_IMAGE
    diff_frame_count++;
#endif

#ifdef OUTPUT_FINAL_DIFF
    diff_final_frame_count++;
#endif

    cv::Mat total_seam_map =
        cv::Mat::ones(scale_height_, scale_width_, CV_8UC1) * 255;
    cv::Mat seam_line = cv::Mat::zeros(scale_height_, scale_width_, CV_8UC1);

    for (int i = 0; i < num_view_; i++) {
        cv::Mat overlap_mask = overlap_masks_[i];
        seam_search(endPts_[i], overlap_mask, diffs_map_[i], total_seam_map,
                    seam_line, i);
    }

    cv::dilate(seam_line, bound_mask_,
               getStructuringElement(cv::MORPH_RECT,
                                     cv::Size(bound_kernel_, bound_kernel_)));
#ifdef OUTPUT_TOTAL_SEAM_MAP
    cv::Mat out =
        cv::Mat::zeros(total_seam_map.rows, total_seam_map.cols, CV_8UC3);
    for (int i = 0; i < out.rows; i++) {
        for (int j = 0; j < out.cols; j++) {
            if (total_seam_map.at<uchar>(i, j) == 0)
                out.at<uchar3>(i, j) = make_uchar3(0, 0, 255);
            else if (total_seam_map.at<uchar>(i, j) == 1)
                out.at<uchar3>(i, j) = make_uchar3(0, 255, 0);
            else if (total_seam_map.at<uchar>(i, j) == 2)
                out.at<uchar3>(i, j) = make_uchar3(255, 0, 0);
            else if (total_seam_map.at<uchar>(i, j) == 3)
                out.at<uchar3>(i, j) = make_uchar3(0, 255, 255);
            else if (total_seam_map.at<uchar>(i, j) == 4)
                out.at<uchar3>(i, j) = make_uchar3(255, 0, 255);
            else if (total_seam_map.at<uchar>(i, j) == 5)
                out.at<uchar3>(i, j) = make_uchar3(255, 255, 0);
        }
    }
    std::filesystem::path total_seam_map_path("./output/total_seam_map/");
    if (!std::filesystem::exists(total_seam_map_path)) {
        std::filesystem::create_directories(total_seam_map_path);
    }
    static int frame_count_total_seam_map = 0;
    total_seam_map_path.append("total_seam_map_" +
                               std::to_string(frame_count_total_seam_map++) +
                               ".png");
    cv::imwrite(total_seam_map_path, out);
#endif

    std::vector<cv::Mat> seam_masks_cpu =
        gen_seam_mask(total_mask_, total_seam_map, num_view_);
    prev_frame_seam = total_seam_map;

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    allocate_seam_masks_GPU(seam_masks_cpu, seam_masks_, down_scale_seam_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    MultiBandBlend_cuda(warped_inputs_, seam_masks_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // draw_icon
    if (show_icon_) {
        draw_icon(this->output_GPUptr(), icon_rgba_, icon_height_, icon_width_,
                  tgt_height_, tgt_width_);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

uchar3* AirViewStitcher::output_GPUptr() {
    return warped_inputs_[1].image;
}

cv::Mat AirViewStitcher::output_CPUMat() {
    uchar3* out_ptr = this->output_GPUptr();

    cv::Mat res = cv::Mat::zeros(tgt_height_, tgt_width_, CV_8UC3);

    cudaMemcpy(res.ptr<uchar3>(0), out_ptr, sizeof(uchar3) * size_tgt_,
               cudaMemcpyDeviceToHost);

    return res;
}
