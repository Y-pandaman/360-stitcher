#include "CameraSender.h"

void CameraSender::setYoloDetector(std::string weight_path) {
    yolo_detect = new yolo_detect::YoloDetect(weight_path, true);
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
    while(true){
        if(gst_receiver == nullptr) {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            return;
        }
        cv::Mat img = gst_receiver->getImageMat();
        if(flag_yolo_detect) {
            img = yolo_detect->detect(img);
        }
        ecal_image_sender.pubImage(img);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        LOG_F(INFO, "cur_frame: %d", cur_frame++);
    }
}

void CameraSender::setGstReceiver(GstReceiver *gst_receiver_) {
//    LOG_F(INFO, "set gst_receiver: %lld", (unsigned long long)gst_receiver_);
    this->gst_receiver = gst_receiver_;
}
