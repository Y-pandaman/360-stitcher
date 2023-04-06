#include "PanoMain.h"

#include <iostream>
#include <stack>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
//#include <cuda.h>
#include <filesystem>
#include "cuda_runtime.h"

#include "Undistorter.h"
#include "airview_stitcher.h"
#include "GstReceiver.h"
#include "yolo_detect.h"
#include <unistd.h>
#include "Intersector.h"
#include "innoreal_timer.hpp"
#include "loguru.hpp"
#include "EcalImageSender.h"
#include "CameraSender.h"

//#define OUTPUT_YOLO_RESULT
//#define OUTPUT_YOLO_ORIGIN_RESULT
//#define OUTPUT_STITCHING_RESULT_PNG
#define OUTPUT_STITCHING_RESULT_VIDEO

//#define USE_GST_INPUT
#define USE_720P
#define USE_VIDEO_INPUT
int panoMain(const std::string &parameters_dir_, bool adjust_rect) {
    std::filesystem::path parameters_dir(parameters_dir_);

//    bool adjust_rect = false;
//    int optc;
//    while ((optc = getopt(argc, argv, "p:a")) != -1) {
//        switch (optc) {
//            case 'p':
//                parameters_dir = std::filesystem::path(optarg);
//                break;
//            case 'a':
//                adjust_rect = true;
//                break;
//            default:
//                break;
//        }
//    }
    std::filesystem::path weight_file_path(parameters_dir);
    weight_file_path.append("weights/best.onnx");
    std::cout << weight_file_path << std::endl;
    yolo_detect::YoloDetect yolo_detect(weight_file_path,
                                        true);
    checkCudaErrors(cudaFree(0));
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float milliseconds = 0;

    std::filesystem::path yamls_dir(parameters_dir);
    yamls_dir.append("yamls");

//    std::filesystem::path data_root_dir = "/home/touch/data/extrin_922/miivii_2";
//    std::filesystem::path data_root_dir = "/home/touch/data/0221_pano_test";
    std::filesystem::path data_root_dir = "/home/touch/data/0313_pano_test/0313_pano_test_0";

    // std::string root_dir = "/home/yons/projects/Stitching/360/";
    // std::string images_dir = root_dir + "922/camera_calib";
    // std::string label_dif = root_dir + "labels/";

    std::vector<int> camera_idx_vec = {0, 3, 2, 6, 1, 7};
//    std::vector<int> camera_idx_vec = {0, 2, 6, 3, 1, 10};

    std::vector<Undistorter> undistorter_vec(camera_idx_vec.size());
    EcalImageSender ecal_image_sender;
    ecal_image_sender.open("overlook_switcher");

    for(int i = 0;i < camera_idx_vec.size();i++) {
        std::filesystem::path yaml_path(yamls_dir);
        yaml_path.append("camera_" + std::to_string(camera_idx_vec[i]) + "_intrin.yaml");
        undistorter_vec[i].loadCameraIntrin(yaml_path.string());
#ifdef USE_720P
        undistorter_vec[i].changeSize(2.0 / 3.0);
#endif
        undistorter_vec[i].getMapForRemapping(1.2, 0.6);
    }

    int src_height = undistorter_vec[0].getNewImageSize().height;
    int src_width = undistorter_vec[0].getNewImageSize().width;
    int HEIGHT = 1080;
    int WIDTH = 1920;
    std::filesystem::path car_rect_yaml(yamls_dir);
    car_rect_yaml.append("car_rect.yaml");
    cv::FileStorage car_rect_fs;
    cv::Rect_<int> car_rect;
    if(!car_rect_fs.open(car_rect_yaml, cv::FileStorage::READ)){
        car_rect.x = WIDTH / 2 - 1000;
        car_rect.y = HEIGHT / 2 - 500;
        car_rect.width = 2000;
        car_rect.height = 1000;
    }
    else {
        car_rect_fs["car_rect"] >> car_rect;
    }
    car_rect_fs.release();

#ifdef OUTPUT_STITCHING_RESULT_VIDEO
    cv::VideoWriter video_writer;
    std::filesystem::path output_video_path("./output/video.avi");
    if(!video_writer.open(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                      cv::Size(WIDTH, HEIGHT), true)) {
        printf("cannot open %s\n", output_video_path.c_str());
        return 0;
    }
#endif

#ifdef USE_VIDEO_INPUT
    std::vector<cv::VideoCapture> video_capture_vec(camera_idx_vec.size());

    for (int i = 0; i < camera_idx_vec.size(); i++) {
        std::filesystem::path video_path(data_root_dir);
//        video_path.append("camera_video_" + std::to_string(camera_idx_vec[i]) + ".avi");
        video_path.append("decode_video_" + std::to_string(camera_idx_vec[i]) + ".avi");
        if (!video_capture_vec[i].open(video_path)) {
            printf("open video error %s\n", video_path.c_str());
        }
    }
#endif

//    use gst receiver
#ifdef USE_GST_INPUT
//    std::vector<int> camera_idx_vec = {0, 3, 2, 6, 1, 7};
    std::vector <std::string> gst_strs = {
            "udpsrc port=5002 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
            "udpsrc port=5000 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
            "udpsrc port=5003 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
            "udpsrc port=5005 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
            "udpsrc port=5001 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
            "udpsrc port=5004 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
    };
//    std::vector <std::string> gst_strs = {
//            "udpsrc port=5002 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink async=false sync=false",
//            "udpsrc port=5000 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink async=false sync=false",
//            "udpsrc port=5003 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink async=false sync=false",
//            "udpsrc port=5005 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink async=false sync=false",
//            "udpsrc port=5001 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink async=false sync=false",
//            "udpsrc port=5004 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink async=false sync=false",
//    };
//    std::vector <std::string> gst_strs = {
//            "udpsrc port=5002 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=BGR ! appsink sync=false",
//            "udpsrc port=5000 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=BGR ! appsink sync=false",
//            "udpsrc port=5003 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=BGR ! appsink sync=false",
//            "udpsrc port=5005 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=BGR ! appsink sync=false",
//            "udpsrc port=5001 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=BGR ! appsink sync=false",
//            "udpsrc port=5004 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw,format=BGR ! appsink sync=false",
//    };
    std::vector <GstReceiver> gst_receivers(camera_idx_vec.size());
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
#endif
    std::vector<std::string> original_ecal_topic_str{
        "left",
        "right",
        "back"
    };
    std::vector<int> original_camera_id_in_vec{2, 0, 4};

#ifdef USE_GST
    std::vector<CameraSender> camera_sender_vec(original_ecal_topic_str.size());
    for(int i = 0; i < original_camera_id_in_vec.size(); i++) {
        std::filesystem::path yaml_path(yamls_dir);
        yaml_path.append("camera_" + std::to_string(camera_idx_vec[original_camera_id_in_vec[i]])
                         + "_intrin.yaml");
//        undistorter_vec[i].loadCameraIntrin(yaml_path.string());
#ifndef USE_720P
        gst_receivers[original_camera_id_in_vec[i]].setUndistorter(yaml_path, 1.0, 0.0, false);
#else
        gst_receivers[original_camera_id_in_vec[i]].setUndistorter(yaml_path, 1.0, 0.0, true);
#endif
        camera_sender_vec[i].setGstReceiver(&gst_receivers[original_camera_id_in_vec[i]]);
        camera_sender_vec[i].setYoloDetector(weight_file_path);
        camera_sender_vec[i].setEcalTopic(original_ecal_topic_str[i]);
        camera_sender_vec[i].startEcalSend();
    }
#endif

    AirViewStitcher *As = new AirViewStitcher(camera_idx_vec.size(),
                                              src_height, src_width, HEIGHT, WIDTH, 2);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 读取 mask 和 homo
    std::vector<cv::Mat> masks(camera_idx_vec.size());
    std::vector<cv::Mat> Hs(camera_idx_vec.size());
    for (int i = 0; i < camera_idx_vec.size(); ++i) {
        cv::Mat mask_image;
        undistorter_vec[i].getMask(mask_image);
        cv::Mat H = cv::Mat::eye(3, 3, CV_64F); // Creating distortion matrix
        std::filesystem::path yaml_path(yamls_dir);
        yaml_path.append("camera_" + std::to_string(camera_idx_vec[i]) + "_homography.yaml");
        cv::FileStorage fs;
        fs.open(yaml_path, cv::FileStorage::READ);
        fs["H"] >> H;
        fs.release();
        H.convertTo(H, CV_32F);

        if (mask_image.rows != src_height) {
            float scale = 1.0f * src_width / mask_image.cols;
            std::cout << "mask size " + std::to_string(camera_idx_vec[i]) + ": " << mask_image.size() << std::endl;
            cv::resize(mask_image, mask_image, cv::Size(src_width, src_height));
            std::cout << "resize " << i << " " << " scale " << scale << " " << camera_idx_vec[i] << std::endl;
            H.at<float>(0, 0) /= scale;
            H.at<float>(0, 1) /= scale;
            H.at<float>(1, 0) /= scale;
            H.at<float>(1, 1) /= scale;
            H.at<float>(2, 0) /= scale;
            H.at<float>(2, 1) /= scale;
        }
        masks[i] = mask_image;
        Hs[i] = H;
    }

#if 0
    std::filesystem::path changed_Hs_fs("video_changed_Hs.yaml");
    cv::FileStorage fs;
    if(!fs.open(changed_Hs_fs, cv::FileStorage::WRITE)) {
        printf("cannot open %s\n", changed_Hs_fs.c_str());
    }
    for(int i = 0;i < Hs.size();i++){
        fs.write("H_" + std::to_string(camera_idx_vec[i]), Hs[i]);
        fs.write("undistort_K_" + std::to_string(camera_idx_vec[i]), undistorter_vec[i].getNewK());
        fs.write("undistort_D_" + std::to_string(camera_idx_vec[i]), undistorter_vec[i].getNewD());
    }
    fs.release();
#endif

    As->init(masks, Hs);
    As->setIcon("car.png", 0.3f);

    std::vector<cv::Mat> input_img_vec(camera_idx_vec.size());
    std::vector<cv::Mat> undistorted_img_vec(camera_idx_vec.size());
    std::vector<std::vector<BBox>> bboxs(camera_idx_vec.size());

    innoreal::InnoRealTimer total_timer, step_timer;

    for(int frame_count = 0;; frame_count++){
        total_timer.TimeStart();
        LOG_F(INFO, "frame_count = %d", frame_count);
        bool video_done_flag = false;

        step_timer.TimeStart();
#ifdef USE_VIDEO_INPUT
        for (int i = 0; i < camera_idx_vec.size(); i++) {
            video_done_flag = !video_capture_vec[i].read(input_img_vec[i]);
            if(video_done_flag) break;
        }
        if(video_done_flag) break;
#endif  // USE_VIDEO_INPUT

#ifdef OUTPUT_YOLO_ORIGIN_RESULT
        for (int i = 0;i < camera_idx_vec.size();i++) {
            cv::Mat tp = input_img_vec[i].clone();
            yolo_detect.detect(tp);
            std::filesystem::path yolo_origin_result_path("./output/yolo_origin_result/");
            yolo_origin_result_path.append(std::to_string(camera_idx_vec[i]));
            if(!std::filesystem::exists(yolo_origin_result_path)) {
                std::filesystem::create_directories(yolo_origin_result_path);
            }
            yolo_origin_result_path.append(std::to_string(camera_idx_vec[i]) + "-" + std::to_string(frame_count) + ".png");
            cv::imwrite(yolo_origin_result_path, tp);
        }
        continue;
#endif
#ifdef USE_GST_INPUT
        step_timer.TimeStart();
        for(int i = 0;i < camera_idx_vec.size();i++){
            input_img_vec[i] = gst_receivers[i].getImageMat();
//            std::cout << "image original size: " << input_img_vec[i].size() << std::endl;
//            cv::imshow("origin_image" + std::to_string(i), input_img_vec[i]);
//            cv::imwrite("./output/720/720_origin_image_" + std::to_string(frame_count) + "_" + std::to_string(i) + ".png", input_img_vec[i]);
        }
        step_timer.TimeEnd();
        LOG_F(INFO, "gst receiver cost %f ms", step_timer.TimeGap_in_ms());
#endif
        step_timer.TimeEnd();
        LOG_F(INFO, "get input images time: %fms", step_timer.TimeGap_in_ms());

#ifdef USE_720P
        step_timer.TimeStart();
        for(int i = 0;i < video_capture_vec.size();i++) {
            if(input_img_vec[i].rows != 720)
               cv::resize(input_img_vec[i], input_img_vec[i], cv::Size(1280, 720));
        }
        step_timer.TimeEnd();
        LOG_F(INFO, "(x)Input 1080 resize to 720 time: %fms.", step_timer.TimeGap_in_ms());
#endif  // USE_720P

        step_timer.TimeStart();
        for (int i = 0; i < input_img_vec.size(); i++) {
//            printf("undistortImage() %d ..\n", i);
            undistorter_vec[i].undistortImage(input_img_vec[i], undistorted_img_vec[i]);
//            printf("undistortImage() %d done\n", i);
//#ifndef USE_720P
//            cv::imwrite("./output/1080/1080_undistorted_image_" + std::to_string(frame_count) + "_" + std::to_string(i) + ".png", undistorted_img_vec[i]);
//#else
//            cv::imwrite("./output/720/720_undistorted_image_" + std::to_string(frame_count) + "_" + std::to_string(i) + ".png", undistorted_img_vec[i]);
//#endif
//            cv::imshow("undistorted_image" + std::to_string(i), undistorted_img_vec[i]);
            if (undistorted_img_vec[i].rows != src_height) {
                cv::resize(undistorted_img_vec[i], undistorted_img_vec[i], cv::Size(src_width, src_height));
            }
        }
        step_timer.TimeEnd();
        LOG_F(INFO, "undistort input image time: %fms", step_timer.TimeGap_in_ms());
        step_timer.TimeStart();
        for(int i = 0;i < input_img_vec.size();i++){
            struct_yolo_result yolo_result = yolo_detect.detect_bbox(undistorted_img_vec[i]);
#ifdef OUTPUT_YOLO_RESULT
            cv::Mat output_img = undistorted_img_vec[i].clone();
            yolo_detect.detect(output_img);
            std::filesystem::path detect_result_path(data_root_dir);
            detect_result_path.append("detected_camera_" + std::to_string(camera_idx_vec[i]));
            if(!std::filesystem::exists(detect_result_path)){
                std::filesystem::create_directories(detect_result_path);
            }
            detect_result_path.append(std::to_string(camera_idx_vec[i]) + "-" + std::to_string(frame_count) + ".png");
            cv::imwrite(detect_result_path, output_img);
            continue;
#endif
            std::vector<BBox> temp;
            for (int j = 0; j < yolo_result.result.size(); j++) {
                temp.emplace_back(BBox(yolo_result.result[j].box, i));
            }
            bboxs[i] = temp;
        }
#ifdef OUTPUT_YOLO_RESULT
        continue;
#endif
        step_timer.TimeEnd();
        LOG_F(INFO, "yolo_detect time: %fms", step_timer.TimeGap_in_ms());

//        cudaEventRecord(start);
        std::vector<std::vector<BBox>> temp(camera_idx_vec.size());
        step_timer.TimeStart();
        As->feed(undistorted_img_vec, temp);
        step_timer.TimeEnd();
        LOG_F(INFO, "As->feed: %fms", step_timer.TimeGap_in_ms());
//        LOG_F(INFO, "as->feed(imgs, bboxs) done");

//        cudaEventRecord(stop);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&milliseconds, start, stop);
        step_timer.TimeStart();
        cv::Mat rgb = As->output_CPUMat();
        step_timer.TimeEnd();
        LOG_F(INFO, "As->output_CPUMat(): %fms", step_timer.TimeGap_in_ms());
//        LOG_F(INFO, "feed to get output CPUMat: %fms", milliseconds);
        while(adjust_rect){
            cv::Mat rgb_clone = rgb.clone();
            cv::rectangle(rgb_clone, car_rect, cv::Scalar(255, 0, 0), 2);
            cv::resize(rgb_clone, rgb_clone, cv::Size(1920, 1080));
            cv::imshow("adjust_rect", rgb_clone);
            int c = cv::waitKey(0);
            int delta = 10;
            switch (c){
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
                    if(!car_rect_fs.open(car_rect_yaml, cv::FileStorage::WRITE)){
                        printf("cannot save car_rect_fs\n");
                    }
//                    car_rect_fs.write("car_rect", car_rect);
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

#if 0
        // test intersector::intersectRects
        float3 test_ps[4];
        test_ps[0] = make_float3(800, 400, 0);
        test_ps[1] = make_float3(1300, 400, 0);
        test_ps[2] = make_float3(1200, 700, 0);
        test_ps[3] = make_float3(800, 700, 0);
        for(int i = 0;i < 4;i++){
            cv::line(rgb, cv::Point(test_ps[i].x, test_ps[i].y), cv::Point(test_ps[(i + 1) % 4].x, test_ps[(i + 1) % 4].y), cv::Scalar(255,255,255));
        }
        if(Intersector::intersectRects(test_ps, car_rect)){
            intersect_warning = true;
            car_rect_color = cv::Scalar(0, 0, 255);
        }
#else
        step_timer.TimeStart();
        for(int i = 0;i < bboxs.size();i++){
            if(intersect_warning) break;
            for(int j = 0; j < bboxs[i].size();j++){
                float3 warped_ps[4];
                if (!bboxs[i][j].get_remapped_points(Hs[i], warped_ps)) continue;
                if(Intersector::intersectRects(warped_ps, car_rect)){
                    intersect_warning = true;
                    car_rect_color = cv::Scalar(0, 0, 255);
                    break;
                }
            }
        }
        step_timer.TimeEnd();
        LOG_F(INFO, "remap box + intersect cost: %fms", step_timer.TimeGap_in_ms());
#endif
        step_timer.TimeStart();
        cv::rectangle(rgb, car_rect, car_rect_color, 2);

//        rgb(cv::Rect(rgb.cols / 4, rgb.rows / 4,
//                     rgb.cols / 2, rgb.rows / 2)).copyTo(rgb);

//        cv::imshow("result img", rgb);
//        cv::waitKey(1);
//        cv::imwrite("result_" + std::to_string(frame_count) + ".png", rgb);
        ecal_image_sender.pubImage(rgb);
        step_timer.TimeEnd();
        LOG_F(INFO, "draw rectangle + pub image cost: %fms", step_timer.TimeGap_in_ms());

        std::cout << "result_image_size: " << rgb.size() << std::endl;
#ifdef OUTPUT_STITCHING_RESULT_VIDEO
        cv::imshow("result_image", rgb);
        cv::waitKey(1);
        video_writer.write(rgb);
#endif

#ifdef OUTPUT_STITCHING_RESULT_PNG
        std::filesystem::path image_result_path("./output/stitching_result_png/");
        if(!std::filesystem::exists(image_result_path)){
            std::filesystem::create_directories(image_result_path);
        }
        image_result_path.append("stitching_result_" + std::to_string(frame_count) + ".png");
        cv::imwrite(image_result_path, rgb);
#endif
        total_timer.TimeEnd();
        LOG_F(INFO, "total time cost: %fms", total_timer.TimeGap_in_ms());
        LOG_F(INFO, "===============================================");
    }

#if 0
    for(int k=0; k<585; k++)
    {
        std::vector<cv::Mat> input_imgs(camera_idx_vec.size());
        std::vector<std::vector<BBox>> bboxs(camera_idx_vec.size());
        for (int i = 0; i < camera_idx_vec.size(); ++i) {
            char name[512];
            sprintf(name, (images_dir + "/%d-%d.png").c_str(), camera_idx_vec[i], k);
            std::cout << name << std::endl;

            cv::Mat img = cv::imread(name);

            if (img.rows != src_height)
            {
                cv::resize(img, img, cv::Size(src_width, src_height));
            }

            input_imgs[i] = img;

            std::vector<BBox> temp;
#if 0
            // char text_name[512];
            // sprintf(text_name, (label_dif + "/%d-%d.txt").c_str(), camera_idx_vec[i], k);

            // std::ifstream infile;
            // infile.open(text_name, std::ios::in);
            // if (!infile.is_open())
            // {
            // 	std::cout << text_name << " 读取失败" << std::endl;
            // }else{
            // 	std::cout << text_name << " 读取成功" << std::endl;
            // 	char data[512];
            // 	while (infile >> data && !infile.eof())
            // 	{
            // 		float4 info;
            // 		std::cout << data << std::endl;
            // 		infile >> info.x >> info.y >> info.z >> info.w;
            // 		std::cout << info.x << " " << info.y << " " << info.z << " " << info.w  << std::endl;
            // 		temp.emplace_back(BBox(info,i, src_height, src_width));
            // 	}
            // 	infile.close();
            // }
#endif
            bboxs[i] = temp;
        }


        cudaEventRecord(start);
        As->feed(input_imgs, bboxs);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout <<  "Total time " <<  milliseconds << " ms" << std::endl;
        cv::Mat rgb = As->output_CPUMat();
        cv::imwrite("blend_"+std::to_string(k)+".png", rgb);
        v_out.write(rgb);
    }

    v_out.release();
#endif

    delete As;

    return 0;
}
