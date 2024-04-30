/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:52:03
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-04-30 19:59:00
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */

#ifndef INC_360CODE_PANOMAIN_H
#define INC_360CODE_PANOMAIN_H

#include "CameraSender.h"
#include "Config.h"
#include "EcalImageSender.h"
#include "GstReceiver.h"
#include "Intersector.h"
#include "Undistorter.h"
#include "airview_stitcher.h"
#include <filesystem>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>

int panoMain(const std::string& parameters_dir_ = "./parameters",
             bool adjust_rect                   = false);

#endif   // INC_360CODE_PANOMAIN_H
