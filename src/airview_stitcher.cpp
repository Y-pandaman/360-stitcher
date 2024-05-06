#include "airview_stitcher.h"

/**
 * AirViewStitcher的构造函数：初始化拼接器的参数和分配相关设备内存。
 *
 * @param num_view 拼接的视图数量。
 * @param src_height 源图像的高度。
 * @param src_width 源图像的宽度。
 * @param tgt_height 目标图像的高度。
 * @param tgt_width 目标图像的宽度。
 * @param down_scale_seam 缩小接缝区域的尺度因子。
 */
AirViewStitcher::AirViewStitcher(int num_view, int src_height, int src_width,
                                 int tgt_height, int tgt_width,
                                 int down_scale_seam) {
    // 初始化基本参数
    num_view_   = num_view;
    src_height_ = src_height;
    src_width_  = src_width;
    tgt_height_ = tgt_height;
    tgt_width_  = tgt_width;

    size_src_ = src_height_ * src_width_;
    size_tgt_ = tgt_height_ * tgt_width_;

    down_scale_seam_ = down_scale_seam;

    // 计算缩放后的尺寸参数
    scale_height_ = tgt_height_ / down_scale_seam_;
    scale_width_  = tgt_width_ / down_scale_seam_;
    size_scale_   = scale_height_ * scale_width_;

    // 为每个视图分配内存
    while (num_view > 0) {
        inputs_.emplace_back(src_height, src_width);
        warped_inputs_.emplace_back(tgt_height, tgt_width);

        // 分配接缝掩码的内存
        uchar* seam_mask;

        checkCudaErrors(
            cudaMalloc((void**)&seam_mask, sizeof(uchar) * size_tgt_));
        seam_masks_.emplace_back(seam_mask);
        // 分配单应性矩阵的内存
        float* H;
        checkCudaErrors(cudaMalloc((void**)&H, sizeof(float) * 9));
        Hs_.emplace_back(H);

        // 分配网格坐标的内存
        float2* grid;
        checkCudaErrors(cudaMalloc((void**)&grid, sizeof(float2) * size_tgt_));
        grids_.emplace_back(grid);

        // 分配缩放后RGB图像的内存
        uchar3* scaled_rgb;
        checkCudaErrors(
            cudaMalloc((void**)&scaled_rgb, sizeof(uchar3) * size_scale_));
        scaled_rgbs_.emplace_back(scaled_rgb);

        // 分配差异图像的内存
        ushort* diff;
        checkCudaErrors(
            cudaMalloc((void**)&diff, sizeof(ushort) * size_scale_));
        diffs_.emplace_back(diff);
        checkCudaErrors(
            cudaMalloc((void**)&diff, sizeof(ushort) * size_scale_));
        temp_diffs_.emplace_back(diff);

        num_view--;
    }
    // 确保所有CUDA操作都已正确完成
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("[ERROR] Create AirViewStitcher failed !!! \n");
    } else {
        LOG_F(INFO, "Create AirViewStitcher successfully!  num_view: %ld",
              num_view_);
    }
}

AirViewStitcher::~AirViewStitcher() {
    printf("[TODO] Not finished yet\n");
    printf("Destroyed AirViewStitcher\n");
}

/**
 * 设置图标。
 * 该函数从指定路径加载图标图像，并将其调整为给定的宽度和高度。然后，它将图标信息存储在类成员变量中，以供后续使用。
 *
 * @param path    图标文件的路径。
 * @param new_width  要调整的图标的新宽度。
 * @param new_height 要调整的图标的新高度。
 *
 * 注：该函数不返回任何值，但它会修改类的内部状态。
 */
void AirViewStitcher::setIcon(std::string path, int new_width, int new_height) {
    // 从路径读取图标图像，保持图像的原始通道数不变
    icon_      = cv::imread(path, cv::IMREAD_UNCHANGED);
    show_icon_ = true;   // 标记为显示图标
    // 记录日志信息，包括图标的原始尺寸和通道数
    LOG_F(INFO, "Load icon: %d x %d x %d", icon_.cols, icon_.rows,
          icon_.channels());

    // 设置图标的目标高度和宽度
    icon_height_ = new_height;
    icon_width_  = new_width;

    // 计算图标总像素数
    size_icon_ = icon_height_ * icon_width_;

    // 调整图标大小以匹配指定的宽度和高度
    cv::resize(icon_, icon_, cv::Size(icon_width_, icon_height_));

    // 在GPU内存上分配空间以存储调整后的图标
    cudaMalloc((void**)&icon_rgba_, sizeof(uchar4) * size_icon_);
    // 将图标数据从主机内存复制到设备内存
    cudaMemcpy(icon_rgba_, icon_.ptr<uchar4>(0), sizeof(uchar4) * size_icon_,
               cudaMemcpyHostToDevice);
}

/**
 * 初始化AirViewStitcher对象。
 * 此函数对输入的掩码和变换矩阵进行处理，包括变换、缩放、侵蚀、阈值处理等操作，以准备进行图像拼接。
 *
 * @param input_masks 输入的视图掩码，为cv::Mat类型的vector，尺寸与视图数一致。
 * @param input_Hs 输入的视图变换矩阵，为cv::Mat类型的vector，尺寸与视图数一致。
 */
void AirViewStitcher::init(std::vector<cv::Mat> input_masks,
                           std::vector<cv::Mat> input_Hs) {
    // 断言输入掩码和变换矩阵的数量与视图数一致
    assert(input_masks.size() == num_view_);
    assert(input_Hs.size() == num_view_);

    // 为存储缩放后的变形掩码初始化容器
    scaled_warped_masks_.resize(num_view_);

    // 遍历每个视图进行处理
    for (uint64_t i = 0; i < num_view_; i++) {
        // 存储变换矩阵
        // Hs_forward_.emplace_back(input_Hs[i]);
        // 将掩码数据从主机内存拷贝到设备内存
        checkCudaErrors(
            cudaMemcpy(inputs_[i].mask, input_masks[i].ptr<uchar>(0),
                       sizeof(uchar) * size_src_, cudaMemcpyHostToDevice));
        // 计算变换矩阵的逆矩阵并拷贝到设备内存
        cv::Mat H = input_Hs[i].inv();
        checkCudaErrors(cudaMemcpy(Hs_[i], H.ptr<float>(0), sizeof(float) * 9,
                                   cudaMemcpyHostToDevice));

        // 计算网格并进行栅格化采样
        // 仿射变换
        compute_grid(Hs_[i], grids_[i], tgt_height_, tgt_width_);
        // 双线性插值
        grid_sample(grids_[i], inputs_[i].mask, warped_inputs_[i].mask,
                    src_height_, src_width_, tgt_height_, tgt_width_);

        // 分配设备内存并拷贝变形后的掩码到主机内存
        uchar* _mask;
        checkCudaErrors(cudaHostAlloc((void**)&_mask, sizeof(uchar) * size_tgt_,
                                      cudaHostAllocDefault));
        checkCudaErrors(cudaMemcpy(_mask, warped_inputs_[i].mask,
                                   sizeof(uchar) * size_tgt_,
                                   cudaMemcpyDeviceToHost));
        // 创建OpenCV的Mat对象以存储掩码
        scaled_warped_masks_[i] =
            cv::Mat(tgt_height_, tgt_width_, CV_8UC1, _mask);

        // 对掩码进行缩放
        cv::resize(scaled_warped_masks_[i], scaled_warped_masks_[i],
                   cv::Size(scale_width_, scale_height_));
        // 掩码图像腐蚀
        cv::erode(scaled_warped_masks_[i], scaled_warped_masks_[i],
                  cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));
        // 掩码图像二值化
        cv::threshold(scaled_warped_masks_[i], scaled_warped_masks_[i], 128,
                      255, cv::THRESH_BINARY);
        // 保存处理后的掩码图像
        cv::imwrite("warped_mask" + std::to_string(i) + ".png",
                    scaled_warped_masks_[i]);
    }

    // 初始化用于视图间重叠区域处理的容器
    overlap_masks_.resize(num_view_);
    endPts_.resize(num_view_);
    diffs_map_.resize(num_view_);
    // 初始化总掩码和边界掩码
    total_mask_ = cv::Mat::zeros(scale_height_, scale_width_, CV_8UC1);
    bound_mask_ = cv::Mat::ones(scale_height_, scale_width_, CV_8UC1) * 255;

    // 计算视图间的重叠区域和更新总掩码
    for (uint64_t i = 0; i < num_view_; i++) {
        int j = (i + 1) % num_view_;

        overlap_masks_[i] = scaled_warped_masks_[i] & scaled_warped_masks_[j];
        total_mask_       = total_mask_ | scaled_warped_masks_[i];
        diffs_map_[i] = cv::Mat::zeros(scale_height_, scale_width_, CV_16UC1);
    }

    // 搜索每个重叠区域的起始和结束点
    for (uint64_t i = 0; i < num_view_; i++) {
        int j = (i + 1) % num_view_;
        endPts_[i] =
            decide_start_end(overlap_masks_[i], scaled_warped_masks_[i],
                             scaled_warped_masks_[j]);
        LOG_F(INFO, "%ld start (%d, %d) end (%d, %d)", i, endPts_[i].x,
              endPts_[i].y, endPts_[i].z, endPts_[i].w);
    }

    // 对总掩码进行侵蚀处理以减少边界错误
    cv::erode(total_mask_, total_mask_,
              cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11)));

    // 确保所有CUDA操作都已正确完成，若有错误则打印错误信息
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("[ERROR] Init AirViewStitcher failed !!! \n");
    } else {
        LOG_F(INFO, "Init AirViewStitcher successfully! num_view: %ld",
              num_view_);
    }
}

/**
 * 向AirViewStitcher实例提供输入图像和对应的边界框信息，并进行图像处理，包括图像的仿射变换、金字塔下采样、差异计算以及最终的图像融合。
 *
 * @param input_img 包含所有视图原图的向量，每个视图的图像为cv::Mat类型。
 * @param bboxs
 * 包含所有物体识别边界框信息的向量，每个视图的边界框信息为std::vector<BBox>类型。
 */
void AirViewStitcher::feed(std::vector<cv::Mat> input_img,
                           std::vector<std::vector<BBox>> bboxs) {
    // 确保输入图像和边界框数量与预期一致
    assert(input_img.size() == num_view_);
    assert(bboxs.size() == num_view_);

    // 遍历所有视图，执行图像预处理步骤
    for (uint64_t i = 0; i < num_view_; i++) {
        // 将图像数据从主机内存复制到设备内存
        checkCudaErrors(
            cudaMemcpy(inputs_[i].image, input_img[i].ptr<uchar3>(0),
                       sizeof(uchar3) * size_src_, cudaMemcpyHostToDevice));
        // 使用grid_sample函数对图像进行仿射变换
        grid_sample(grids_[i], inputs_[i].image, warped_inputs_[i].image,
                    src_height_, src_width_, tgt_height_, tgt_width_);
        // 确保cuda操作完成
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        // 对warp后的图像进行金字塔下采样
        pyramid_downsample(warped_inputs_[i], down_scale_seam_,
                           scaled_rgbs_[i]);
    }

    // 用于存储上一帧的缝合线
    static cv::Mat prev_frame_seam;

    // 计算所有视图两两之间的差异图像
    for (uint64_t i = 0; i < num_view_; i++) {
        int j = (i + 1) % num_view_;
        compute_diff(scaled_rgbs_[i], scaled_rgbs_[j], diffs_[i],
                     temp_diffs_[i], scale_height_, scale_width_);

        // 更新相邻两个图像中人物区域的差异值
        std::vector<BBox> boundingboxs_i = bboxs[i];
        std::vector<BBox> boundingboxs_j = bboxs[j];
        for (uint64_t k = 0; k < boundingboxs_i.size(); k++) {
            update_diff_in_person_area(diffs_[i], grids_[i], down_scale_seam_,
                                       boundingboxs_i[k].data, scale_height_,
                                       scale_width_);
        }
        for (uint64_t k = 0; k < boundingboxs_j.size(); k++) {
            update_diff_in_person_area(diffs_[i], grids_[j], down_scale_seam_,
                                       boundingboxs_j[k].data, scale_height_,
                                       scale_width_);
        }
        // 使用上一帧的缝合线更新差异图像
        if (!prev_frame_seam.empty()) {
            update_diff_use_seam(diffs_[i], prev_frame_seam, scale_height_,
                                 scale_width_);
        }
        // 将差异图像数据从设备内存拷贝到主机内存
        cudaMemcpy(diffs_map_[i].ptr<ushort>(0), diffs_[i],
                   sizeof(ushort) * size_scale_, cudaMemcpyDeviceToHost);
    }

    // 计算所有视图的缝合线
    cv::Mat total_seam_map =
        cv::Mat::ones(scale_height_, scale_width_, CV_8UC1) * 255;
    cv::Mat seam_line = cv::Mat::zeros(scale_height_, scale_width_, CV_8UC1);
    for (uint64_t i = 0; i < num_view_; i++) {
        cv::Mat overlap_mask = overlap_masks_[i];
        seam_search(endPts_[i], overlap_mask, diffs_map_[i], total_seam_map,
                    seam_line, i);
    }

    // 对缝合线进行膨胀处理，生成边界掩膜
    cv::dilate(seam_line, bound_mask_,
               cv::getStructuringElement(
                   cv::MORPH_RECT, cv::Size(bound_kernel_, bound_kernel_)));

    // 生成每个视图的接缝掩膜
    // total_mask_: 全部视图的掩膜图像
    // total_seam_map: 全部视图的接缝图
    // num_view_: 视图的数量
    // 返回值: 包含每个视图接缝掩膜的向量
    std::vector<cv::Mat> seam_masks_cpu =
        gen_seam_mask(total_mask_, total_seam_map, num_view_);
    prev_frame_seam = total_seam_map;

    // 确保所有CUDA操作都已完成，检查是否有错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 分配GPU内存以存储接缝掩膜
    // seam_masks_cpu: 在CPU上生成的接缝掩膜向量
    // seam_masks_: 存储在GPU上的接缝掩膜
    // down_scale_seam_: 接缝掩膜的降尺度版本
    allocate_seam_masks_GPU(seam_masks_cpu, seam_masks_, down_scale_seam_);

    // 确保所有CUDA操作都已完成，检查是否有错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 使用多带混合方法融合图像
    // warped_inputs_: 变形后的输入图像集合
    // seam_masks_: 接缝掩膜集合，用于指定融合过程中要忽略的像素
    MultiBandBlend_cuda(warped_inputs_, seam_masks_);
    // 确保所有CUDA操作都已完成，检查是否有错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 绘制图标到输出图像上
    if (show_icon_) {
        draw_icon(this->output_GPUptr(), icon_rgba_, icon_height_, icon_width_,
                  tgt_height_, tgt_width_);
    }

    // 确保所有CUDA操作完成
    checkCudaErrors(cudaDeviceSynchronize());
    // 检查最后一个CUDA操作是否有错误
    checkCudaErrors(cudaGetLastError());
}

/**
 * @brief 获取经过处理的图像数据的GPU指针
 *
 * 该函数不接受参数。
 *
 * @return uchar3* 返回一个指向经过处理的图像数据的GPU内存指针。
 */
uchar3* AirViewStitcher::output_GPUptr() {
    // 返回第二个输入图像（warped_inputs_数组中的第二个元素）的图像数据指针
    return warped_inputs_[1].image;
}

/**
 * 从GPU内存中获取处理后的图像，并转换为OpenCV的Mat格式。
 *
 * @return cv::Mat 返回一个大小为tgt_height_ x tgt_width_的3通道8位彩色图像。
 */
cv::Mat AirViewStitcher::output_CPUMat() {
    // 从GPU内存获取输出图像的指针
    uchar3* out_ptr = this->output_GPUptr();

    // 初始化一个空的OpenCV Mat对象，用于存放结果图像
    cv::Mat res = cv::Mat::zeros(tgt_height_, tgt_width_, CV_8UC3);

    // 将GPU内存中的图像数据拷贝到CPU内存的Mat对象中
    cudaMemcpy(res.ptr<uchar3>(0), out_ptr, sizeof(uchar3) * size_tgt_,
               cudaMemcpyDeviceToHost);

    // 返回包含图像数据的Mat对象
    return res;
}
