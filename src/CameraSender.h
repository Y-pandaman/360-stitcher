//
// Created by xcmg on 23-4-1.
//

#ifndef PANO_CODE_CAMERASENDER_H
#define PANO_CODE_CAMERASENDER_H


#include "yolo_detect.h"
#include "EcalImageSender.h"
#include <atomic>
#include <thread>
#include <GstReceiver.h>
#include "loguru.hpp"

class CameraSender {
public:
    void setYoloDetector(std::string weight_path);
    void setEcalTopic(std::string ecal_topic_str);
    void setGstReceiver(GstReceiver *gst_receiver_);
    void startEcalSend();
private:
    void mainWorker();
    EcalImageSender ecal_image_sender;
    std::thread *main_worker_thread = nullptr;
    std::atomic_bool flag_yolo_detect = false;
    yolo_detect::YoloDetect *yolo_detect;
    GstReceiver *gst_receiver = nullptr;
};


#endif //PANO_CODE_CAMERASENDER_H
