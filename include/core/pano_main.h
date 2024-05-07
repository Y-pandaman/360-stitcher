/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:52:03
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-06 20:49:15
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */

#ifndef INC_360CODE_PANOMAIN_H
#define INC_360CODE_PANOMAIN_H

#include "core/airview_stitcher.h"
#include "stage/camera_sender.h"
#include "stage/ecal_image_sender.h"
#include "stage/gst_receiver.h"
#include "stage/undistorter.h"
#include "yolo/yolo_detect.h"
#include "yolo/yolov5trt_det.h"
#include "util/Intersector.h"
#include "util/config.h"
#include <filesystem>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

int panoMain(const std::string& parameters_dir_ = "./parameters",
             bool adjust_rect                   = false);

#endif   // INC_360CODE_PANOMAIN_H
