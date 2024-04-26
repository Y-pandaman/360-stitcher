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
    std::shared_ptr<eCAL::protobuf::CPublisher<proto_messages::OpencvImage>>
        m_pub_image;
};

#endif   // CYLINDER_STITCHER_ECALIMAGESENDER_H
