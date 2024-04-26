#include "Config.h"

Config::Config() {
    load_config_file("../parameters/yamls/config.yaml");
}

bool Config::load_config_file(const std::string& file_path) {
    cv::FileStorage fs;
    if (!fs.open(file_path, cv::FileStorage::READ)) {
        printf("cannot open config file %s\n", file_path.c_str());
        exit(0);
    }

    fs["icon_new_height"] >> icon_new_height;
    fs["icon_new_width"] >> icon_new_width;
    fs["send_undistorted_image"] >> send_undistorted_image;

    fs["use_reproj"] >> use_reproj;

    fs["final_crop_w_left"] >> final_crop_w_left;
    fs["final_crop_w_right"] >> final_crop_w_right;
    fs["final_crop_h_top"] >> final_crop_h_top;
    fs["final_crop_h_bottom"] >> final_crop_h_bottom;

    fs.release();
    return true;
}
