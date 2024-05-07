#include "stage/undistorter.h"

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

/**
 * 该函数用于对图像进行去畸变处理。
 *
 * @param input_image 输入图像，需要进行去畸变处理的原始图像。
 * @param output_image 输出图像，经过去畸变处理后的图像。
 * @return 成功返回true，如果无法初始化映射或映射获取失败则返回false。
 */
bool Undistorter::undistortImage(cv::Mat input_image, cv::Mat& output_image) {
    // 检查映射是否已经初始化，如果未初始化则尝试获取映射
    if (!map_inited) {
        bool flag = getMapForRemapping();   // 尝试获取用于映射的参数
        if (!flag)   // 如果获取失败，则返回false
            return false;
    }
    // 使用remap函数对图像进行去畸变处理
    cv::remap(input_image, output_image, map1, map2, cv::INTER_LINEAR);
    return true;
}
/**
 * 获取经过畸变矫正的掩码图像。
 *
 * @param out_mask
 * 引用，用于存储输出的掩码图像。掩码图像为8位单通道图像，初始值为全1矩阵。
 * @return 成功返回true，如果未初始化映射或掩码未生成则返回false。
 */
bool Undistorter::getMask(cv::Mat& out_mask) {
    // 检查映射是否已初始化
    if (!map_inited)
        return false;

    // 如果掩码未生成，生成并存储掩码
    if (!mask_generated) {
        mask =
            cv::Mat::ones(input_image_size, CV_8UC1) * 255;   // 创建初始全1掩码
        cv::remap(mask, mask, map1, map2,
                  cv::INTER_LINEAR);   // 应用映射进行畸变矫正
        mask_generated = true;         // 标记掩码为已生成
    }

    out_mask = mask.clone();   // 复制掩码到输出参数
    return true;
}

// 根据缩放的图像大小确定相机内参，乘以对应的图像比例
void Undistorter::changeSize(float factor) {
    K *= factor;
    K.at<double>(2, 2) = 1;

    input_image_size.height *= factor;
    input_image_size.width *= factor;
}

// 将轨迹线叠加在后视图上
void FishToCylProj::stitch_project_to_cyn() {
    // 检查cuda错误
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 如果存在背景跟踪图像，将背景跟踪图像与当前视图混合；否则，将当前视图投影到圆柱体上。
    if (m_sub_back_track != nullptr) {
        cv::Mat back_track_image = getBackTrackImage();   // 获取当前帧

        if (!back_track_image.empty()) {
            setExtraImage(back_track_image);   // 将图像放到GPU上
            // 如果存在back_track_image，使用CUDA加速将额外视图图像混合到屏幕上。
            BlendExtraViewToScreen_cuda(view_.image, extra_view.image,
                                        view_.width, view_.height, 1.0);
        }
    }
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
    // 订阅Back_Track_Image数据
    m_sub_back_track =
        std::make_shared<eCAL::protobuf::CSubscriber<xcmg_proto::OpencvImage>>(
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

// 获取叠加后的图像
cv::Mat FishToCylProj::getProjectedImage() {
    cv::Mat img, mask;
    // 将GPU上的图像数据和掩码数据复制到CPU内存中的img和mask变量
    view_.toCPU(img, mask);
    cv::Mat result_img = img.clone();
    return result_img;
}

// 将图像放到GPU上
void FishToCylProj::setExtraImage(cv::Mat extra_img) {
    extra_view.height = extra_img.rows;
    extra_view.width  = extra_img.cols;
    extra_view.mask   = nullptr;

    // 检查extra_view_buffer是否为空指针。如果是，表示尚未为extra_img的设备（GPU）内存分配空间
    if (extra_view_buffer == nullptr) {
        // 使用cudaMalloc函数为GPU设备内存分配空间，用于存储extra_img
        checkCudaErrors(
            cudaMalloc((void**)&extra_view_buffer,
                       extra_img.rows * extra_img.cols * sizeof(uchar3)));
    }

    // 将CPU内存中的extra_img数据复制到之前分配的GPU内存中
    checkCudaErrors(
        cudaMemcpy(extra_view_buffer, extra_img.data,
                   extra_view.width * extra_view.height * sizeof(uchar3),
                   cudaMemcpyHostToDevice));

    extra_view.image = (uchar3*)extra_view_buffer;
}
