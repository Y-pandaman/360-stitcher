#include "Undistorter.h"

// 加载相机内参
bool Undistorter::loadCameraIntrin(const std::string& fs_path) {
    cv::FileStorage fs;
    if (!fs.open(fs_path, cv::FileStorage::Mode::READ)) {
        printf("cannot open fs file %s\n", fs_path.c_str());
        return false;
    }
    fs["K"] >> K;
    fs["D"] >> D;
    fs["image_size"] >> input_image_size;

    // getMapForRemapping(1.0, 0.0);
    map_inited = true;
    return true;
}

// 估计新的相机内参矩阵,无畸变后的
bool Undistorter::getMapForRemapping(float new_size_factor, float balance) {
    if (K.empty() || D.empty()) {
        printf("K & D empty, cannot get map for remapping\n");
        return false;
    }
    cv::Mat eye_mat = cv::Mat::eye(3, 3, CV_32F);
    this->new_image_size =
        cv::Size(this->input_image_size.width * new_size_factor,
                 this->input_image_size.height * new_size_factor);
    //    std::cout << "new_image_size: " << new_image_size << std::endl;
    // 为去畸变和图像校正估计新的相机矩阵
    // 用来调节去畸变之后的视野的，去畸变之后一般视野会变小，是否需要保持原来的视野。
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        K, D, input_image_size, eye_mat, new_K, balance, new_image_size);
    // 计算去畸变和图像校正的映射，如果D为空，则使用零失真；如果R为空，则使用单位矩阵
    // 此函数为了提高算法运行速度
    cv::fisheye::initUndistortRectifyMap(K, D, eye_mat, new_K, new_image_size,
                                         CV_16SC2, this->map1, this->map2);
    new_D = D;
    //    std::cout << "map1.size: " << map1.size() << std::endl;
    return true;
}

bool Undistorter::undistortImage(cv::Mat input_image, cv::Mat& output_image) {
    if (!map_inited) {
        bool flag = getMapForRemapping();
        if (!flag)
            return false;
    }
    cv::remap(input_image, output_image, map1, map2, cv::INTER_LINEAR);
    return true;
}

bool Undistorter::getMask(cv::Mat& out_mask) {
    if (!map_inited)
        return false;
    if (!mask_generated) {
        mask = cv::Mat::ones(input_image_size, CV_8UC1) * 255;
        cv::remap(mask, mask, map1, map2, cv::INTER_LINEAR);
        mask_generated = true;
    }

    out_mask = mask.clone();
    return true;
}

// 根据缩放的图像大小确定相机内参，乘以对应的图像比例
void Undistorter::changeSize(float factor) {
    K *= factor;
    K.at<double>(2, 2) = 1;

    input_image_size.height *= factor;
    input_image_size.width *= factor;
}

// 将图像投影到圆柱面上
void FishToCylProj::stitch_project_to_cyn(int time) {
    if (cyl_ == nullptr) {
        float r = 1000.0;   // 圆柱半径
        // 创建圆柱投影对象
        cyl_ = new CylinderGPU_stilib(view_.camera.R, view_.camera.C, r);
        cyl_image_width_  = view_.width * 1.5;    // 圆柱体图像的宽度
        cyl_image_height_ = view_.height * 1.2;   // 圆柱体图像的高度
        // 根据col_grid_num_（列网格数）调整圆柱体图像的高度/宽度，确保它是行/列网格数的整数倍。
        cyl_image_width_ =
            ((cyl_image_width_ - 1) / col_grid_num_ + 1) * col_grid_num_;
        cyl_image_height_ =
            ((cyl_image_height_ - 1) / row_grid_num_ + 1) * row_grid_num_;

        //创建一个圆柱体图像对象CylinderImageGPU_stilib，使用调整后的宽度和高度进行初始化
        cyl_image_ =
            CylinderImageGPU_stilib(cyl_image_height_, cyl_image_width_);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    if (m_sub_back_track != nullptr) {
        cv::Mat back_track_image = getBackTrackImage();

        if (!back_track_image.empty()) {
            setExtraImage(back_track_image);
            BlendExtraViewToScreen_cuda(view_.image, extra_view.image,
                                        view_.width, view_.height, 1.0);
        }
    } else {
        projToCylinderImage_cuda(view_, cyl_image_, *cyl_, cyl_image_width_,
                                 cyl_image_height_);
    }

#ifdef OUTPUT_CYL_IMAGE
    cv::Mat rgb_0, mask_0;
    static int cyl_image_count = 0;
    cyl_image_.toCPU(rgb_0, mask_0);
    cv::imwrite("./output/out_image/cyl_image_" +
                    std::to_string(cyl_image_count) + ".png",
                rgb_0);
    cyl_image_count++;
#endif
}

FishToCylProj::FishToCylProj(const Undistorter& undistorter) {
    Eigen::Matrix4f extrin     = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f extrin_inv = extrin.inverse();

    std::vector<float> R, T, C;
    float *d_R, *d_T, *d_C;

    R.emplace_back(extrin(0, 0));
    R.emplace_back(extrin(0, 1));
    R.emplace_back(extrin(0, 2));
    R.emplace_back(extrin(1, 0));
    R.emplace_back(extrin(1, 1));
    R.emplace_back(extrin(1, 2));
    R.emplace_back(extrin(2, 0));
    R.emplace_back(extrin(2, 1));
    R.emplace_back(extrin(2, 2));
    checkCudaErrors(cudaMalloc((void**)&d_R, 9 * sizeof(float)));
    checkCudaErrors(
        cudaMemcpy(d_R, R.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

    T.emplace_back(extrin(0, 3));
    T.emplace_back(extrin(1, 3));
    T.emplace_back(extrin(2, 3));
    checkCudaErrors(cudaMalloc((void**)&d_T, 3 * sizeof(float)));
    checkCudaErrors(
        cudaMemcpy(d_T, T.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

    C.emplace_back(extrin_inv(0, 3));
    C.emplace_back(extrin_inv(1, 3));
    C.emplace_back(extrin_inv(2, 3));
    checkCudaErrors(cudaMalloc((void**)&d_C, 3 * sizeof(float)));
    checkCudaErrors(
        cudaMemcpy(d_C, C.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

    cv::Mat matK        = undistorter.getK();
    cv::Mat matD        = undistorter.getD();
    cv::Size input_size = undistorter.getInputSize();
    view_.camera        = PinholeCameraGPU_stilib(
        matK.at<double>(0, 0), matK.at<double>(1, 1), matK.at<double>(0, 2),
        matK.at<double>(1, 2), matD.at<double>(0, 0), matD.at<double>(1, 0),
        matD.at<double>(2, 0), matD.at<double>(3, 0), d_R, d_T, d_C);

    view_.height = input_size.height;
    view_.width  = input_size.width;

    checkCudaErrors(cudaMalloc((void**)&view_.image,
                               sizeof(uchar3) * view_.height * view_.width));

    if (!eCAL::IsInitialized())
        eCAL::Initialize();
    eCAL::Util::EnableLoopback(true);

    m_sub_back_track = std::make_shared<
        eCAL::protobuf::CSubscriber<proto_messages::OpencvImage>>(
        "Back_Track_Image");

    auto cb = std::bind(&FishToCylProj::ecalBackTrackImageCallBack, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5);

    m_sub_back_track->AddReceiveCallback(cb);
}

// 将Mat图像复制到GPU内存中
void FishToCylProj::setImage(cv::Mat input_img) {
    assert(input_img.rows == view_.height && input_img.cols == view_.width);
    // cudaMemcpy: CPU和GPU之间的内存复制
    // cudaMemcpyHostToDevice: CPU内存到GPU内存
    checkCudaErrors(cudaMemcpy(view_.image, input_img.data,
                               view_.width * view_.height * sizeof(uchar3),
                               cudaMemcpyHostToDevice));
}

cv::Mat FishToCylProj::getProjectedImage() {
    cv::Mat img, mask;
    float factor = 0.0;
    view_.toCPU(img, mask);
    cv::Mat result_img = img.clone();
    return result_img;
}

void FishToCylProj::setExtraImageCuda(float* image_cuda, int width,
                                      int height) {
    extra_view.width  = width;
    extra_view.height = height;
    extra_view.mask   = nullptr;

    int block = 128, grid = (height * width + block - 1) / block;

    if (extra_view_buffer == nullptr)
        checkCudaErrors(cudaMalloc((void**)&extra_view_buffer,
                                   width * height * sizeof(uchar3)));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    ConvertRGBAF2RGBU_host(image_cuda, (uchar3*)extra_view_buffer, width,
                           height, grid, block);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    extra_view.image = (uchar3*)extra_view_buffer;
}

void FishToCylProj::setExtraImage(cv::Mat extra_img) {
    extra_view.height = extra_img.rows;
    extra_view.width  = extra_img.cols;
    extra_view.mask   = nullptr;

    if (extra_view_buffer == nullptr)
        checkCudaErrors(
            cudaMalloc((void**)&extra_view_buffer,
                       extra_img.rows * extra_img.cols * sizeof(uchar3)));

    checkCudaErrors(
        cudaMemcpy(extra_view_buffer, extra_img.data,
                   extra_view.width * extra_view.height * sizeof(uchar3),
                   cudaMemcpyHostToDevice));

    extra_view.image = (uchar3*)extra_view_buffer;
}
