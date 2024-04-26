#include "PanoMain.h"
#include "CameraSender.h"
#include "EcalImageSender.h"
#include "GstReceiver.h"
#include "Intersector.h"
#include "Undistorter.h"
#include "airview_stitcher.h"
#include "cuda_runtime.h"
#include "innoreal_timer.hpp"
#include "loguru.hpp"
#include "yolo_detect.h"
#include <filesystem>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <stack>
#include <unistd.h>

//#define OUTPUT_YOLO_RESULT
//#define OUTPUT_YOLO_ORIGIN_RESULT
//#define OUTPUT_STITCHING_RESULT_PNG
#define OUTPUT_STITCHING_RESULT_VIDEO

// #define USE_GST_INPUT
// #define RESEND_ORIGINAL_IMAGE
#define USE_VIDEO_INPUT
#define USE_720P

#include "Config.h"

static Config config;

int panoMain(const std::string& parameters_dir_, bool adjust_rect) {
    std::filesystem::path parameters_dir(parameters_dir_);

    // 启动yolo识别器
    std::filesystem::path weight_file_path(parameters_dir);
    weight_file_path.append("weights/best.onnx");
    yolo_detect::YoloDetect yolo_detect(weight_file_path, true);
    checkCudaErrors(cudaFree(0));
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float milliseconds = 0;

    // 加载内参文件
    std::filesystem::path yamls_dir(parameters_dir);
    yamls_dir.append("yamls");

    // 录制的图像文件
    std::filesystem::path data_root_dir = "/home/bdca/camera_video";

    //    std::vector<int> camera_idx_vec = {3, 6, 2, 1};
    //    std::vector<int> camera_idx_vec = {0, 3, 6, 2, 7, 1};
    // 相机索引
    // 左后-左前-前-右前-右后-后
    std::vector<int> camera_idx_vec = {1, 3, 0, 4, 2, 5};

    // 去畸变
    std::vector<Undistorter> undistorter_vec(camera_idx_vec.size());
    // 初始化ecal
    EcalImageSender ecal_image_sender;
    ecal_image_sender.open("overlook_switcher");

    for (int i = 0; i < camera_idx_vec.size(); i++) {
        std::filesystem::path yaml_path(yamls_dir);
        yaml_path.append("camera_" + std::to_string(camera_idx_vec[i]) +
                         "_intrin.yaml");
        undistorter_vec[i].loadCameraIntrin(
            yaml_path.string());   // 加载相机内参
#ifdef USE_720P
        undistorter_vec[i].changeSize(2.0 / 3.0);   // 改变内参K值
#endif
        undistorter_vec[i].getMapForRemapping(1.2, 0.1);   // 计算变换矩阵
    }

    int src_height = undistorter_vec[0].getNewImageSize().height;
    int src_width  = undistorter_vec[0].getNewImageSize().width;
    int HEIGHT     = 720;
    int WIDTH      = 1280;
    //    int HEIGHT = 1080;
    //    int WIDTH = 1920;
    std::filesystem::path car_rect_yaml(yamls_dir);
    car_rect_yaml.append("car_rect.yaml");   // 读取车辆位置
    cv::FileStorage car_rect_fs;
    cv::Rect_<int> car_rect;
    if (!car_rect_fs.open(car_rect_yaml, cv::FileStorage::READ)) {
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
    cv::VideoWriter video_writer;
    std::filesystem::path output_video_path("../output/video.avi");
    if (!video_writer.open(output_video_path,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                           cv::Size(WIDTH, HEIGHT), true)) {
        printf("cannot open %s\n", output_video_path.c_str());
        return 0;
    }
#endif

// 使用图像文件
#ifdef USE_VIDEO_INPUT
    std::vector<cv::VideoCapture> video_capture_vec(camera_idx_vec.size());

    for (int i = 0; i < camera_idx_vec.size(); i++) {
        std::filesystem::path video_path(data_root_dir);
        video_path.append("camera_video_" + std::to_string(camera_idx_vec[i]) +
                          ".avi");
        //        video_path.append("decode_video_" +
        //        std::to_string(camera_idx_vec[i]) + ".avi");
        if (!video_capture_vec[i].open(video_path)) {
            printf("open video error %s\n", video_path.c_str());
        }
    }
#endif

// 使用gst码流
#ifdef USE_GST_INPUT
    //    std::vector<int> camera_idx_vec = {0, 3, 6, 2, 7, 1};
    //    std::vector<int> camera_idx_vec = {3, 6, 2, 1};
    //    std::vector<int> camera_idx_vec = {1,3,0,4,2,5};
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
    for (int i = 0; i < camera_idx_vec.size(); i++) {
        printf("initialize VideoCapture %d...\n", i);
        if (gst_receivers[i].initialize(gst_strs[i], 2)) {
            printf("initialize VideoCapture %d done\n", i);
        }
    }
    for (int i = 0; i < camera_idx_vec.size(); i++) {
        if (gst_receivers[i].startReceiveWorker()) {
            printf("start gst_receiver %d done\n", i);
        }
    }
#endif   // USE_GST_INPUT

#ifdef USE_GST_INPUT
#ifdef RESEND_ORIGINAL_IMAGE
    std::vector<std::string> original_ecal_topic_str {"left", "right", "back"};
    std::vector<int> original_camera_id_in_vec {1, 3, 5};
    std::vector<CameraSender> camera_sender_vec(original_ecal_topic_str.size());
    for (int i = 0; i < original_camera_id_in_vec.size(); i++) {
        std::filesystem::path yaml_path(yamls_dir);
        // 加载内参
        yaml_path.append(
            "camera_" +
            std::to_string(camera_idx_vec[original_camera_id_in_vec[i]]) +
            "_intrin.yaml");
//        undistorter_vec[i].loadCameraIntrin(yaml_path.string());
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
                true);   // 去畸变的标志位，无用（默认不去畸变）
        }
        camera_sender_vec[i].setGstReceiver(
            &gst_receivers[original_camera_id_in_vec[i]]);
        //        camera_sender_vec[i].setYoloDetector(weight_file_path);
        camera_sender_vec[i].setEcalTopic(original_ecal_topic_str[i]);
        camera_sender_vec[i].startEcalSend();   // 发送ecal
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

    int window_size = 30;
    std::queue<float> time_window;
    float time_sum = 0;

    for (int i = 0; i < camera_idx_vec.size(); ++i) {
        cv::Mat mask_image;
        undistorter_vec[i].getMask(mask_image);

        float factor = 0.3;
        if (camera_idx_vec[i] == 1) {
            mask_image(cv::Range(0, mask_image.rows),
                       cv::Range(0, mask_image.cols * factor))
                .setTo(0);
        }
        //        if(camera_idx_vec[i] == 3){
        //            mask_image(cv::Range(0, mask_image.rows), cv::Range(0,
        //            mask_image.cols * factor)).setTo(0);
        //        }

        if (camera_idx_vec[i] == 4) {
            mask_image(cv::Range(0, mask_image.rows),
                       cv::Range(0, mask_image.cols * factor))
                .setTo(0);
        }
        //        if(camera_idx_vec[i] == 2) {
        //            mask_image(cv::Range(0, mask_image.rows), cv::Range(0,
        //            mask_image.cols * (factor))).setTo(0);
        //        }

        //        cv::imshow("mask " + std::to_string(camera_idx_vec[i]),
        //        mask_image); cv::waitKey(0);
        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);   // Creating distortion matrix
        std::filesystem::path yaml_path(yamls_dir);
        yaml_path.append("camera_" + std::to_string(camera_idx_vec[i]) +
                         "_homography.yaml");   // 读取单应性矩阵
        cv::FileStorage fs;
        fs.open(yaml_path, cv::FileStorage::READ);
        fs["H"] >> H;
        fs.release();
        H.convertTo(H, CV_32F);

        if (mask_image.rows != src_height) {
            float scale = 1.0f * src_width / mask_image.cols;
            std::cout << "mask size " + std::to_string(camera_idx_vec[i]) + ": "
                      << mask_image.size() << std::endl;
            cv::resize(mask_image, mask_image, cv::Size(src_width, src_height));
            std::cout << "resize " << i << " "
                      << " scale " << scale << " " << camera_idx_vec[i]
                      << std::endl;
            H.at<float>(0, 0) /= scale;
            H.at<float>(0, 1) /= scale;
            H.at<float>(1, 0) /= scale;
            H.at<float>(1, 1) /= scale;
            H.at<float>(2, 0) /= scale;
            H.at<float>(2, 1) /= scale;
        }
        masks[i] = mask_image;
        Hs[i]    = H;
    }

    As->init(masks, Hs);
    As->setIcon("../assets/car.png", config.icon_new_width,
                config.icon_new_height);   // 图像中粘贴车图像

    std::vector<cv::Mat> input_img_vec(camera_idx_vec.size());
    std::vector<cv::Mat> undistorted_img_vec(camera_idx_vec.size());

    innoreal::InnoRealTimer total_timer, step_timer;   // 计时器

    std::vector<std::vector<BBox>> bboxs(camera_idx_vec.size());

    for (int frame_count = 0;; frame_count++) {
        total_timer.TimeStart();
        //        LOG_F(INFO, "frame_count = %d", frame_count);
        bool video_done_flag = false;

        step_timer.TimeStart();
        // 使用图像文件
#ifdef USE_VIDEO_INPUT
        for (int i = 0; i < camera_idx_vec.size(); i++) {
            video_done_flag = !video_capture_vec[i].read(input_img_vec[i]);
            if (video_done_flag) {
                video_capture_vec[i].set(cv::CAP_PROP_POS_FRAMES, 0);
                video_done_flag = !video_capture_vec[i].read(input_img_vec[i]);
            }
        }
        if (video_done_flag)
            break;
#endif   // USE_VIDEO_INPUT

// 输出yolo识别结果
#ifdef OUTPUT_YOLO_ORIGIN_RESULT
        for (int i = 0; i < camera_idx_vec.size(); i++) {
            cv::Mat tp = input_img_vec[i].clone();
            yolo_detect.detect(tp);
            std::filesystem::path yolo_origin_result_path(
                "./output/yolo_origin_result/");
            yolo_origin_result_path.append(std::to_string(camera_idx_vec[i]));
            if (!std::filesystem::exists(yolo_origin_result_path)) {
                std::filesystem::create_directories(yolo_origin_result_path);
            }
            yolo_origin_result_path.append(std::to_string(camera_idx_vec[i]) +
                                           "-" + std::to_string(frame_count) +
                                           ".png");
            cv::imwrite(yolo_origin_result_path, tp);
        }
        continue;
#endif

// 使用gst数据
#ifdef USE_GST_INPUT
        step_timer.TimeStart();
        for (int i = 0; i < camera_idx_vec.size(); i++) {
            input_img_vec[i] = gst_receivers[i].getImageMat();
            //            std::cout << "image original size: " <<
            //            input_img_vec[i].size() << std::endl;
            //            cv::imshow("origin_image" + std::to_string(i),
            //            input_img_vec[i]);
            //            cv::imwrite("./output/720/720_origin_image_" +
            //            std::to_string(frame_count) + "_" + std::to_string(i)
            //            +
            //            ".png", input_img_vec[i]);
        }
        step_timer.TimeEnd();
//        LOG_F(INFO, "gst receiver cost %f ms", step_timer.TimeGap_in_ms());
#endif

// 重置图像尺寸
#ifdef USE_720P
        step_timer.TimeStart();
        for (int i = 0; i < input_img_vec.size(); i++) {
            if (input_img_vec[i].rows != 720)
                cv::resize(input_img_vec[i], input_img_vec[i],
                           cv::Size(1280, 720));
        }
        step_timer.TimeEnd();

#endif   // USE_720P

        step_timer.TimeStart();
        for (int i = 0; i < input_img_vec.size(); i++) {
            undistorter_vec[i].undistortImage(input_img_vec[i],
                                              undistorted_img_vec[i]);   // 变换
            if (undistorted_img_vec[i].rows != src_height) {
                cv::resize(undistorted_img_vec[i], undistorted_img_vec[i],
                           cv::Size(src_width, src_height));
            }
        }
        step_timer.TimeEnd();
        //        LOG_F(INFO, "undistort input image time: %fms",
        //        step_timer.TimeGap_in_ms());

        for (int i = 0; i < input_img_vec.size(); i++) {
            struct_yolo_result yolo_result =
                yolo_detect.detect_bbox(undistorted_img_vec[i]);
            // 输出yolo结果
#ifdef OUTPUT_YOLO_RESULT
            cv::Mat output_img = undistorted_img_vec[i].clone();
            yolo_detect.detect(output_img);
            std::filesystem::path detect_result_path(data_root_dir);
            detect_result_path.append("detected_camera_" +
                                      std::to_string(camera_idx_vec[i]));
            if (!std::filesystem::exists(detect_result_path)) {
                std::filesystem::create_directories(detect_result_path);
            }
            detect_result_path.append(std::to_string(camera_idx_vec[i]) + "-" +
                                      std::to_string(frame_count) + ".png");
            cv::imwrite(detect_result_path, output_img);
            continue;
#endif
            std::vector<BBox> temp;
            for (int j = 0; j < yolo_result.result.size(); j++) {
                temp.emplace_back(BBox(yolo_result.result[j].box, i));
            }
            bboxs[i] = temp;
        }

        std::vector<std::vector<BBox>> temp(camera_idx_vec.size());
        step_timer.TimeStart();
        As->feed(undistorted_img_vec, bboxs);
        step_timer.TimeEnd();

        step_timer.TimeStart();
        cv::Mat rgb = As->output_CPUMat();
        step_timer.TimeEnd();

        // 调整贴图的位置
        while (adjust_rect) {
            cv::Mat rgb_clone = rgb.clone();
            cv::rectangle(rgb_clone, car_rect, cv::Scalar(255, 0, 0), 2);
            cv::resize(rgb_clone, rgb_clone, cv::Size(1920, 1080));
            cv::imshow("adjust_rect", rgb_clone);
            int c     = cv::waitKey(0);
            int delta = 10;
            switch (c) {
            case 'w':
                car_rect.y -= delta;
                break;
            case 's':
                car_rect.y += delta;
                break;
            case 'a':
                car_rect.x -= delta;
                break;
            case 'd':
                car_rect.x += delta;
                break;
            case 'r':
                car_rect.height += delta;
                break;
            case 'f':
                car_rect.height -= delta;
                break;
            case 't':
                car_rect.width += delta;
                break;
            case 'g':
                car_rect.width -= delta;
                break;
            case 'o':
                adjust_rect = false;
                if (!car_rect_fs.open(car_rect_yaml, cv::FileStorage::WRITE)) {
                    printf("cannot save car_rect_fs\n");
                }
                car_rect_fs << "car_rect" << car_rect;
                car_rect_fs.release();
                cv::destroyAllWindows();
                break;
            default:
                break;
            }
        }
        // check whether bbox intersect car_rect
        cv::Scalar car_rect_color(255, 0, 0);
        bool intersect_warning = false;

        step_timer.TimeStart();
        innoreal::InnoRealTimer detail_timer;

        for (int i = 0; i < bboxs.size(); i++) {
            if (intersect_warning)
                break;
            //            LOG_F(INFO, "bbox num = %d\n", bboxs[i].size());
            for (int j = 0; j < bboxs[i].size(); j++) {
                float3 warped_ps[4];
                detail_timer.TimeStart();
                if (!bboxs[i][j].get_remapped_points(Hs[i], warped_ps))
                    continue;
                detail_timer.TimeEnd();

                detail_timer.TimeStart();
                if (Intersector::intersectRects(warped_ps, car_rect)) {
                    intersect_warning = true;
                    car_rect_color    = cv::Scalar(0, 0, 255);
                    break;
                }
                detail_timer.TimeEnd();
            }
        }
        step_timer.TimeEnd();

        step_timer.TimeStart();

        // 贴图
        cv::rectangle(rgb, car_rect, car_rect_color, 2);

        rgb(cv::Rect(config.final_crop_w_left, config.final_crop_h_top,
                     rgb.cols - config.final_crop_w_left -
                         config.final_crop_w_right,
                     rgb.rows - config.final_crop_h_top -
                         config.final_crop_h_bottom))
            .copyTo(rgb);

        ecal_image_sender.pubImage(rgb);
        step_timer.TimeEnd();
        std::cout << "result_image_size: " << rgb.size() << std::endl;

#ifdef OUTPUT_STITCHING_RESULT_VIDEO
        cv::imshow("result_image", rgb);
        cv::waitKey(1);
        video_writer.write(rgb);
#endif

#ifdef OUTPUT_STITCHING_RESULT_PNG
        std::filesystem::path image_result_path(
            "./output/stitching_result_png/");
        if (!std::filesystem::exists(image_result_path)) {
            std::filesystem::create_directories(image_result_path);
        }
        image_result_path.append("stitching_result_" +
                                 std::to_string(frame_count) + ".png");
        cv::imwrite(image_result_path, rgb);
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
        LOG_F(INFO, "mean time cost: %fms in %d frames", mean_time,
              window_size);
    }
    delete As;

    return 0;
}
