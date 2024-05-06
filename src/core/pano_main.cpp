#include "core/pano_main.h"

// #define OUTPUT_STITCHING_RESULT_VIDEO

#define USE_GST_INPUT
#define RESEND_ORIGINAL_IMAGE
// #define USE_VIDEO_INPUT
#define USE_720P

static Config config;

int panoMain(const std::string& parameters_dir_, bool adjust_rect) {
    std::filesystem::path parameters_dir(parameters_dir_);

    // 启动yolo识别器
    std::filesystem::path weight_file_path(parameters_dir);
    weight_file_path.append("weights/best.onnx");
    yolo_detect::YoloDetect yolo_detect(weight_file_path, true);   // 初始化yolo
    checkCudaErrors(cudaFree(0));
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // 加载内参文件
    std::filesystem::path yamls_dir(parameters_dir);
    yamls_dir.append("yamls");   // 相机的内参，单应性矩阵

    // 录制的图像文件路径
    std::filesystem::path data_root_dir = "/home/bdca/camera_video1";

    // 相机索引
    // 左后-左前-前-右前-右后-后
    std::vector<int> camera_idx_vec = {1, 3, 0, 4, 2, 5};

    // 声明去畸变对象
    std::vector<Undistorter> undistorter_vec(camera_idx_vec.size());
    // 初始化ecal
    EcalImageSender ecal_image_sender;
    ecal_image_sender.open("overlook_switcher");   // 定义一个ecal发布器

    for (uint64_t i = 0; i < camera_idx_vec.size(); i++) {
        std::filesystem::path yaml_path(yamls_dir);
        yaml_path.append("camera_" + std::to_string(camera_idx_vec[i]) +
                         "_intrin.yaml");
        undistorter_vec[i].loadCameraIntrin(
            yaml_path.string());   // 加载相机内参，去畸变参数
#ifdef USE_720P
        undistorter_vec[i].changeSize(2.0 / 3.0);   // 根据图像大小改变内参K值
#endif
        undistorter_vec[i].getMapForRemapping(1.2, 0.1);   // 计算变换矩阵
    }

    int src_height = undistorter_vec[0].getNewImageSize().height;
    int src_width  = undistorter_vec[0].getNewImageSize().width;
    int HEIGHT     = 720;
    int WIDTH      = 1280;
    // 读取车辆外范围位置
    std::filesystem::path car_rect_yaml(yamls_dir);
    car_rect_yaml.append("car_rect.yaml");
    cv::FileStorage car_rect_fs;
    cv::Rect_<int> car_rect;
    if (!car_rect_fs.open(car_rect_yaml, cv::FileStorage::READ)) {
        LOG_F(WARNING, "Failed to open car_rect, %s", car_rect_yaml.c_str());
        car_rect.x      = WIDTH / 2 - 1000;
        car_rect.y      = HEIGHT / 2 - 500;
        car_rect.width  = 2000;
        car_rect.height = 1000;
    } else {
        car_rect_fs["car_rect"] >> car_rect;
    }
    car_rect_fs.release();

// 输出拼接结果
#ifdef OUTPUT_STITCHING_RESULT_VIDEO
    // 定义视频写入器
    cv::VideoWriter video_writer;
    std::filesystem::path output_video_path("../output/video.avi");
    if (!video_writer.open(output_video_path,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                           cv::Size(WIDTH, HEIGHT), true)) {
        printf("cannot open %s\n", output_video_path.c_str());
        return 0;
    }
#endif

// 读取图像文件
#ifdef USE_VIDEO_INPUT
    std::vector<cv::VideoCapture> video_capture_vec(camera_idx_vec.size());

    for (uint64_t i = 0; i < camera_idx_vec.size(); i++) {
        std::filesystem::path video_path(data_root_dir);
        video_path.append("camera_video_" + std::to_string(camera_idx_vec[i]) +
                          ".avi");
        if (!video_capture_vec[i].open(video_path)) {
            printf("open video error %s\n", video_path.c_str());
        }
    }
#endif

// 使用gst码流
#ifdef USE_GST_INPUT
    std::vector<std::string> gst_strs = {
        "udpsrc port=5101 "
        "caps=application/"
        "x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)"
        "H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! "
        "queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
        "udpsrc port=5103 "
        "caps=application/"
        "x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)"
        "H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! "
        "queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
        "udpsrc port=5100 "
        "caps=application/"
        "x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)"
        "H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! "
        "queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
        "udpsrc port=5104 "
        "caps=application/"
        "x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)"
        "H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! "
        "queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
        "udpsrc port=5102 "
        "caps=application/"
        "x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)"
        "H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! "
        "queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
        "udpsrc port=5105 "
        "caps=application/"
        "x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)"
        "H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! "
        "queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
    };

    // 接收图像
    std::vector<GstReceiver> gst_receivers(camera_idx_vec.size());
    // 依次初始化码流抓取器
    for (uint64_t i = 0; i < camera_idx_vec.size(); i++) {
        printf("initialize VideoCapture %ld...\n", i);
        if (gst_receivers[i].initialize(gst_strs[i], 2)) {
            printf("initialize VideoCapture %ld done\n", i);
        }
    }
    // 依次抓取视频流
    for (uint64_t i = 0; i < camera_idx_vec.size(); i++) {
        if (gst_receivers[i].startReceiveWorker()) {
            printf("start gst_receiver %ld done\n", i);
        }
    }
#endif   // USE_GST_INPUT

#ifdef USE_GST_INPUT
#ifdef RESEND_ORIGINAL_IMAGE
    std::vector<std::string> original_ecal_topic_str {"left", "right", "back"};
    std::vector<int> original_camera_id_in_vec {1, 3, 5};
    // 初始化图像发布器
    std::vector<CameraSender> camera_sender_vec(original_ecal_topic_str.size());
    for (uint64_t i = 0; i < original_camera_id_in_vec.size(); i++) {
        std::filesystem::path yaml_path(yamls_dir);
        // 加载内参
        yaml_path.append(
            "camera_" +
            std::to_string(camera_idx_vec[original_camera_id_in_vec[i]]) +
            "_intrin.yaml");
        // 获取去畸变的相机矩阵
#ifndef USE_720P
        gst_receivers[original_camera_id_in_vec[i]].setUndistorter(
            yaml_path, 1.0, 0.0, false);
#else
        gst_receivers[original_camera_id_in_vec[i]].setUndistorter(
            yaml_path, 1.0, 0.0, true);
#endif
        if (original_ecal_topic_str[i].compare("back") == 0) {
            camera_sender_vec[i].setUndistorter(
                undistorter_vec[original_camera_id_in_vec[i]],
                true);   // 是否添加倒车辅助线
        }
        camera_sender_vec[i].setGstReceiver(
            &gst_receivers[original_camera_id_in_vec[i]]);
        camera_sender_vec[i].setEcalTopic(
            original_ecal_topic_str[i]);        // 初始化ecal发布器
        camera_sender_vec[i].startEcalSend();   // 发送ecal数据
    }
#endif   // RESEND_ORIGINAL_IMAGE

#endif   // USE_GST_INPUT

    AirViewStitcher* As = new AirViewStitcher(camera_idx_vec.size(), src_height,
                                              src_width, HEIGHT, WIDTH, 2);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 读取 mask 和 homo
    std::vector<cv::Mat> masks(camera_idx_vec.size());
    std::vector<cv::Mat> Hs(camera_idx_vec.size());

    uint64_t window_size = 30;
    std::queue<float> time_window;
    float time_sum = 0;

    for (uint64_t i = 0; i < camera_idx_vec.size(); ++i) {
        cv::Mat mask_image;
        undistorter_vec[i].getMask(mask_image);   // 获取去畸变的掩码图像

        float factor = 0.3;
        if (camera_idx_vec[i] == 1) {
            mask_image(
                cv::Range(0, mask_image.rows),
                cv::Range(mask_image.cols * (1 - factor), mask_image.cols))
                .setTo(0);
        }
        if (camera_idx_vec[i] == 5) {
            mask_image(cv::Range(0, mask_image.rows),
                       cv::Range(0, mask_image.cols * factor))
                .setTo(0);
        }

        if (camera_idx_vec[i] == 4) {
            mask_image(
                cv::Range(0, mask_image.rows),
                cv::Range(mask_image.cols * (1 - factor), mask_image.cols))
                .setTo(0);
        }
        if (camera_idx_vec[i] == 3) {
            mask_image(cv::Range(0, mask_image.rows),
                       cv::Range(0, mask_image.cols * (factor)))
                .setTo(0);
        }
        // 创建畸变矩阵，初始为单位矩阵
        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);

        // 构建YAML文件路径，用于读取单应性矩阵
        std::filesystem::path yaml_path(yamls_dir);
        yaml_path.append("camera_" + std::to_string(camera_idx_vec[i]) +
                         "_homography.yaml");

        // 打开YAML文件并读取单应性矩阵H
        cv::FileStorage fs;
        fs.open(yaml_path, cv::FileStorage::READ);
        fs["H"] >> H;
        fs.release();   // 关闭文件存储

        // 将H矩阵的数据类型转换为32位浮点数
        H.convertTo(H, CV_32F);

        // 检查掩膜图像的行数是否与源图像的高度相同
        if (mask_image.rows != src_height) {
            // 计算缩放比例，以使掩膜图像的宽度与源图像宽度相匹配
            float scale = 1.0f * src_width / mask_image.cols;
            // 打印原始掩膜图像尺寸
            std::cout << "mask size " + std::to_string(camera_idx_vec[i]) + ": "
                      << mask_image.size() << std::endl;
            // 调整掩膜图像的尺寸以匹配源图像的尺寸
            cv::resize(mask_image, mask_image, cv::Size(src_width, src_height));
            // 打印调整后的掩膜图像尺寸和缩放信息
            std::cout << "resize " << i << " "
                      << " scale " << scale << " " << camera_idx_vec[i]
                      << std::endl;
            // 调整变换矩阵H的尺度，以匹配掩膜图像的缩放
            H.at<float>(0, 0) /= scale;
            H.at<float>(0, 1) /= scale;
            H.at<float>(1, 0) /= scale;
            H.at<float>(1, 1) /= scale;
            H.at<float>(2, 0) /= scale;
            H.at<float>(2, 1) /= scale;
        }
        // 将调整后的掩膜图像和变换矩阵添加到对应数组中
        masks[i] = mask_image;
        Hs[i]    = H;
    }

    As->init(masks, Hs);   // 初始化拼接器
    As->setIcon("../assets/car.png", config.icon_new_width,
                config.icon_new_height);   // 图像中粘贴车图像

    std::vector<cv::Mat> input_img_vec(camera_idx_vec.size());
    std::vector<cv::Mat> undistorted_img_vec(camera_idx_vec.size());

    innoreal::InnoRealTimer total_timer, step_timer;   // 计时器

    std::vector<std::vector<BBox>> bboxs(camera_idx_vec.size());

    for (int frame_count = 0;; frame_count++) {
        total_timer.TimeStart();

        step_timer.TimeStart();
        // 使用图像文件
#ifdef USE_VIDEO_INPUT
        bool video_done_flag = false;
        // 遍历所有摄像头索引，尝试从每个摄像头读取一帧图像
        for (uint64_t i = 0; i < camera_idx_vec.size(); i++) {
            // 尝试读取视频帧，如果读取失败（视频结束），则将视频指针重置为第一帧，并再次尝试读取
            video_done_flag = !video_capture_vec[i].read(input_img_vec[i]);
            if (video_done_flag) {
                video_capture_vec[i].set(cv::CAP_PROP_POS_FRAMES, 0);
                video_done_flag = !video_capture_vec[i].read(input_img_vec[i]);
            }
        }
        // 如果所有视频都已结束，则退出循环
        if (video_done_flag)
            break;
#endif   // USE_VIDEO_INPUT

// 使用gst数据
#ifdef USE_GST_INPUT
        for (uint64_t i = 0; i < camera_idx_vec.size(); i++) {
            input_img_vec[i] =
                gst_receivers[i].getImageMat();   // 获取当前帧图像
        }
#endif

// 重置图像尺寸
#ifdef USE_720P
        for (uint64_t i = 0; i < input_img_vec.size(); i++) {
            if (input_img_vec[i].rows != 720)
                cv::resize(input_img_vec[i], input_img_vec[i],
                           cv::Size(1280, 720));
        }

#endif   // USE_720P

        // 遍历输入图像向量，并对每张图像进行去畸变处理
        for (uint64_t i = 0; i < input_img_vec.size(); i++) {
            // 对当前图像进行去畸变处理
            undistorter_vec[i].undistortImage(input_img_vec[i],
                                              undistorted_img_vec[i]);
            // 如果处理后的图像高度与期望高度不一致，则调整图像尺寸
            if (undistorted_img_vec[i].rows != src_height) {
                cv::resize(undistorted_img_vec[i], undistorted_img_vec[i],
                           cv::Size(src_width, src_height));
            }
        }

        // 遍历输入图像向量，对每张图像进行YOLO检测，并处理检测结果
        for (uint64_t i = 0; i < input_img_vec.size(); i++) {
            // 使用YOLO检测算法对当前图像进行物体检测，并获取检测结果
            struct_yolo_result yolo_result =
                yolo_detect.detect_bbox(undistorted_img_vec[i]);

            // 初始化存储检测框的向量
            std::vector<BBox> temp;
            // 遍历当前图像的检测结果，将每个检测框与对应图像的索引封装成BBox对象，并加入向量中
            for (uint64_t j = 0; j < yolo_result.result.size(); j++) {
                temp.emplace_back(BBox(yolo_result.result[j].box, i));
            }
            // 将当前图像的检测框向量存入总的结果向量中
            bboxs[i] = temp;
        }
        std::vector<std::vector<BBox>> temp(camera_idx_vec.size());

        As->feed(undistorted_img_vec, bboxs);   // 执行拼接
        cv::Mat rgb = As->output_CPUMat();   // 带贴图的图像从GPU拷贝到CPU

        /**
         * 调整给定外框（car_rect）的位置，通过键盘输入上下左右及大小调整，
         * 最终将调整后的外框保存到配置文件中。
         * 该过程会在一个窗口中显示调整过程的实时反馈。
         */
        while (adjust_rect) {
            // 克隆原图像，并在图像上绘制当前车外框的位置
            cv::Mat rgb_clone = rgb.clone();
            cv::rectangle(rgb_clone, car_rect, cv::Scalar(255, 0, 0), 2);

            // 将图像调整为1920x1080大小以适应显示
            cv::resize(rgb_clone, rgb_clone, cv::Size(1920, 1080));

            // 显示带有外框绘制的图像，等待用户输入
            cv::imshow("adjust_rect", rgb_clone);

            // 等待用户按键输入，根据按键调整外框位置或大小
            int c     = cv::waitKey(0);
            int delta = 10;
            switch (c) {
            case 'w':
                car_rect.y -= delta;   // 向上移动外框
                break;
            case 's':
                car_rect.y += delta;   // 向下移动外框
                break;
            case 'a':
                car_rect.x -= delta;   // 向左移动外框
                break;
            case 'd':
                car_rect.x += delta;   // 向右移动外框
                break;
            case 'r':
                car_rect.height += delta;   // 增高外框
                break;
            case 'f':
                car_rect.height -= delta;   // 减低外框
                break;
            case 't':
                car_rect.width += delta;   // 增宽外框
                break;
            case 'g':
                car_rect.width -= delta;   // 减窄外框
                break;
            case 'o':
                adjust_rect = false;   // 结束位置调整

                // 保存调整后的外框到配置文件
                if (!car_rect_fs.open(car_rect_yaml, cv::FileStorage::WRITE)) {
                    printf("cannot save car_rect_fs\n");
                }
                car_rect_fs << "car_rect" << car_rect;
                car_rect_fs.release();   // 关闭文件存储

                // 关闭所有显示窗口
                cv::destroyAllWindows();
                break;
            default:
                break;
            }
        }

        // 初始化车辆外框的颜色为蓝色，相交警告标志为false
        cv::Scalar car_rect_color(255, 0, 0);
        bool intersect_warning = false;

        // 遍历所有bounding boxes来检查相交情况
        for (uint64_t i = 0; i < bboxs.size(); i++) {
            // 如果已经检测到相交，则跳出循环
            if (intersect_warning)
                break;
            // 遍历当前bounding box中的所有点
            for (uint64_t j = 0; j < bboxs[i].size(); j++) {
                float3 warped_ps[4];
                // 如果无法获取重映射点，则跳过当前点
                if (!bboxs[i][j].get_remapped_points(Hs[i], warped_ps))
                    continue;

                // 检查当前点是否与车辆矩形相交
                if (Intersector::intersectRects(warped_ps, car_rect)) {
                    // 如果相交，则设置相交警告标志，改变车辆矩形颜色为红色
                    intersect_warning = true;
                    car_rect_color    = cv::Scalar(0, 0, 255);
                    break;   // 退出内层循环，检查下一个bounding box
                }
            }
        }

        // 贴图
        cv::rectangle(rgb, car_rect, car_rect_color, 2);

        rgb(cv::Rect(config.final_crop_w_left, config.final_crop_h_top,
                     rgb.cols - config.final_crop_w_left -
                         config.final_crop_w_right,
                     rgb.rows - config.final_crop_h_top -
                         config.final_crop_h_bottom))
            .copyTo(rgb);

        ecal_image_sender.pubImage(rgb);

#ifdef OUTPUT_STITCHING_RESULT_VIDEO
        cv::imshow("result_image", rgb);
        cv::waitKey(1);
        video_writer.write(rgb);
#endif

        total_timer.TimeEnd();
        float total_time = total_timer.TimeGap_in_ms();
        time_sum += total_time;
        time_window.push(total_time);
        while (time_window.size() > window_size) {
            time_sum -= time_window.front();
            time_window.pop();
        }
        float mean_time = time_sum / time_window.size();
        LOG_F(INFO, "mean time cost: %fms in %ld frames", mean_time,
              window_size);
    }
    delete As;

    return 0;
}