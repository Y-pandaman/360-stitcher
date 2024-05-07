/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:53:47
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-06 11:22:54
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
//
// Created by touch on 22-11-29.
//

#ifndef CYLINDER_STITCHER_ECALIMAGESENDER_H
#define CYLINDER_STITCHER_ECALIMAGESENDER_H

#include "image.pb.h"
#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>
#include <memory>
#include <opencv2/opencv.hpp>

class EcalImageSender {
public:
    EcalImageSender(int argc = 0, char** argv = nullptr);
    ~EcalImageSender();

    void open(const std::string& topic);
    void pubImage(cv::Mat image);

private:
    std::shared_ptr<eCAL::protobuf::CPublisher<xcmg_proto::OpencvImage>>
        m_pub_image;
};

#endif   // CYLINDER_STITCHER_ECALIMAGESENDER_H
