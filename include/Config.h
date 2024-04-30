#ifndef PANO_CODE_CONFIG_H
#define PANO_CODE_CONFIG_H

#include <opencv2/opencv.hpp>
#include <string>

class Config {
public:
    Config();
    bool load_config_file(const std::string& file_path);

    int icon_new_width, icon_new_height;
    int final_crop_w_left;
    int final_crop_w_right;
    int final_crop_h_top;
    int final_crop_h_bottom;

private:
};

#endif   // PANO_CODE_CONFIG_H
