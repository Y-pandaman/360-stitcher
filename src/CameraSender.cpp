/*
 * @Author: 姚潘涛
 * @Date: 2024-04-23 19:16:24
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-06 19:30:51
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "CameraSender.h"

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
            LOG_F(ERROR, "gst_receiver is null");
            return;
        }
        cv::Mat img = gst_receiver->getImageMat();   // 获取当前帧数据
        if (fish_to_cyl_proj != nullptr) {
            assert(fish_to_cyl_proj != nullptr);
            fish_to_cyl_proj->setImage(img);   // 图像从CPU复制到GPU
            // 将辅助线叠加在后视图上
            fish_to_cyl_proj->stitch_project_to_cyn();
            img = fish_to_cyl_proj->getProjectedImage();   // 获取叠加后的图像
            // cv::imshow("back", img);
            // cv::imwrite(std::to_string(cur_frame) + ".png", img);
        }
        ecal_image_sender.pubImage(img);   // 发布图像
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
        fish_to_cyl_proj = new FishToCylProj(undistorter_);
    }
}
