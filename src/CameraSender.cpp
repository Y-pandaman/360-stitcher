/*
 * @Author: 姚潘涛
 * @Date: 2024-04-23 19:16:24
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-04-25 18:32:33
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "CameraSender.h"

void CameraSender::setYoloDetector(std::string weight_path) {
    yolo_detect      = new yolo_detect::YoloDetect(weight_path, true);
    flag_yolo_detect = true;
}

void CameraSender::setEcalTopic(std::string ecal_topic_str) {
    ecal_image_sender.open(ecal_topic_str);
}

void CameraSender::startEcalSend() {
    main_worker_thread = new std::thread(&CameraSender::mainWorker, this);
}

void CameraSender::mainWorker() {
    int cur_frame = 0;
    while (true) {
        if (gst_receiver == nullptr) {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            return;
        }
        cv::Mat img = gst_receiver->getImageMat();
        if (fish_to_cyl_proj != nullptr) {
            assert(fish_to_cyl_proj != nullptr);
            fish_to_cyl_proj->setImage(img);
            fish_to_cyl_proj->stitch_project_to_cyn(cur_frame);
            img = fish_to_cyl_proj->getProjectedImage();
        }
        ecal_image_sender.pubImage(img);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        cur_frame++;
    }
}

void CameraSender::setGstReceiver(GstReceiver* gst_receiver_) {
    this->gst_receiver = gst_receiver_;
}

void CameraSender::setUndistorter(const Undistorter& undistorter_,
                                  bool set_thread_cyl) {
    undistorter = undistorter_;
    if (set_thread_cyl) {
        fish_to_cyl_proj   = new FishToCylProj(undistorter_);
        flag_fishToCylProj = false;
        flag_undistort     = false;
    } else {
        flag_undistort     = false;
        flag_fishToCylProj = false;
    }
}
