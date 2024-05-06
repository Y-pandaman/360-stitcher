/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:44:05
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-06 09:52:12
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */

#ifndef PANO_CODE_CAMERASENDER_H
#define PANO_CODE_CAMERASENDER_H

#include "EcalImageSender.h"
#include "GstReceiver.h"
#include "Undistorter.h"
#include "image.pb.h"
#include "loguru.hpp"
#include "yolo_detect.h"
#include <atomic>
#include <thread>

class CameraSender {
public:
    void setEcalTopic(std::string ecal_topic_str);
    void setGstReceiver(GstReceiver* gst_receiver_);
    void startEcalSend();
    void setUndistorter(const Undistorter& undistorter_,
                        bool set_thread_cyl = false);

private:
    Undistorter undistorter;
    void mainWorker();
    EcalImageSender ecal_image_sender;
    std::thread* main_worker_thread     = nullptr;
    GstReceiver* gst_receiver       = nullptr;
    FishToCylProj* fish_to_cyl_proj = nullptr;
};

#endif   // PANO_CODE_CAMERASENDER_H
