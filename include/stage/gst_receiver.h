/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:53:47
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-06 20:03:27
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */

#ifndef CYLINDER_STITCHER_GSTRECEIVER_H
#define CYLINDER_STITCHER_GSTRECEIVER_H

#include "stage/ecal_image_sender.h"
#include "stage/undistorter.h"
#include "util/loguru.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

class GstReceiver {
public:
    bool initialize(const std::string& url_, int queue_size_ = 2);
    bool startReceiveWorker();
    uchar* getImageData();
    cv::Mat getImageMat();
    ~GstReceiver();

    void setUndistorter(const std::string& intrin_file_path,
                        float new_size_factor = 1.0, float balance = 0.0,
                        bool USE_720P = false);

private:
    int frame_count = 0;
    void receiveWorker();
    std::condition_variable cond_image_queue_not_empty;
    std::atomic_bool stop_flag = false;
    cv::VideoCapture video_capture;
    int queue_size;
    int video_width, video_height;
    std::string video_url;
    cv::Mat* image_queue               = nullptr;
    std::mutex* mutex_on_image_queue   = nullptr;
    std::thread *thread_receive_worker = nullptr,
                *thread_write_worker   = nullptr;

    std::atomic_ullong p_write = 0, p_read = 0;
    enum Status {
        UNINITIALIZED,
        INITIALIZED,
        RUN,
        STOP
    } status = UNINITIALIZED;

    Undistorter undistorter;
};

#endif   // CYLINDER_STITCHER_GSTRECEIVER_H
