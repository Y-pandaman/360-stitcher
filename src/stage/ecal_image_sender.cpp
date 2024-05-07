/*
 * @Author: 姚潘涛
 * @Date: 2024-04-23 19:16:24
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-04-30 19:49:05
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "stage/ecal_image_sender.h"

EcalImageSender::EcalImageSender(int argc, char** argv) { }

EcalImageSender::~EcalImageSender() { }

/**
 * @brief 发布OpenCV图像
 *
 * 此函数将OpenCV图像封装到自定义的protobuf消息中，并通过一个图像发布者发送出去。
 *
 * @param image 要发布的OpenCV图像。图像数据将被复制到protobuf消息中。
 */
void EcalImageSender::pubImage(cv::Mat image) {
    // 创建protobuf消息，用于存储图像信息和数据
    xcmg_proto::OpencvImage message_opencv_image;

    // 设置图像的行数、列数和元素类型
    message_opencv_image.set_rows(image.rows);
    message_opencv_image.set_cols(image.cols);
    message_opencv_image.set_elt_type(image.type());

    // 计算图像数据的大小，并将图像数据复制到protobuf消息中
    size_t data_size = image.rows * image.cols * image.elemSize();
    message_opencv_image.set_mat_data(image.data, data_size);

    // 发送protobuf消息
    m_pub_image->Send(message_opencv_image);
}

/**
 * @brief 初始化并打开一个eCAL图像发送器，用于发送OpenCV图像。
 *
 * @param topic 要发布图像的eCAL主题名称。
 */
void EcalImageSender::open(const std::string& topic) {
    // 检查eCAL是否已经初始化，如果没有则进行初始化
    if (!eCAL::IsInitialized()) {
        eCAL::Initialize();
    }
    // 启用eCAL的环回功能，使得发布的消息可以被本地订阅者接收到
    eCAL::Util::EnableLoopback(true);
    // 创建一个基于指定主题的图像发布器
    m_pub_image =
        std::make_shared<eCAL::protobuf::CPublisher<xcmg_proto::OpencvImage>>(
            topic);
}
