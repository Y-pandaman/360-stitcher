//
// Created by touch on 22-12-13.
//

#ifndef CALIB_GROUND_UNDISTORTER_H
#define CALIB_GROUND_UNDISTORTER_H

#include <opencv2/opencv.hpp>
#include <cstring>
//#include "camera_calib.h"

class Undistorter {
public:
    bool loadCameraIntrin(const std::string &fs_path);
    bool getMapForRemapping(float new_size_factor = 1.0, float balance = 1.0);
    bool undistortImage(cv::Mat input_image, cv::Mat & output_image);
    bool getMask(cv::Mat &out_mask);
    cv::Mat getNewK(){return new_K;}
    cv::Mat getNewD(){return new_D;}
    cv::Size getNewImageSize(){return new_image_size;}

    void changeSize(float factor);
private:
    cv::Mat K, D;
    cv::Size input_image_size;
    cv::Mat new_K, new_D;
    cv::Size new_image_size;
    cv::Mat map1, map2; // map for remapping
    cv::Mat mask;
    bool mask_generated = false;
    bool map_inited = false;
};

#endif //CALIB_GROUND_UNDISTORTER_H
