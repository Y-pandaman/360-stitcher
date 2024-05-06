

#ifndef CALIB_GROUND_UNDISTORTER_H
#define CALIB_GROUND_UNDISTORTER_H

#include "image.pb.h"
#include "innoreal_timer.hpp"
#include "render.cuh"
#include <cstring>
#include <ecal/ecal.h>
#include <ecal/msg/protobuf/subscriber.h>
#include <opencv2/opencv.hpp>

class Undistorter {
public:
    bool loadCameraIntrin(const std::string& fs_path);
    bool getMapForRemapping(float new_size_factor = 1.0, float balance = 1.0);
    bool undistortImage(cv::Mat input_image, cv::Mat& output_image);
    bool getMask(cv::Mat& out_mask);
    cv::Mat getNewK() const {
        return new_K;
    }
    cv::Mat getNewD() const {
        return new_D;
    }
    cv::Size getNewImageSize() const {
        return new_image_size;
    }

    cv::Mat getK() const {
        return K;
    }
    cv::Mat getD() const {
        return D;
    }
    cv::Size getInputSize() const {
        return input_image_size;
    }

    void changeSize(float factor);

private:
    cv::Mat K, D;
    cv::Size input_image_size;
    cv::Mat new_K, new_D;
    cv::Size new_image_size;
    cv::Mat map1, map2;   // map for remapping
    cv::Mat mask;
    bool mask_generated = false;
    bool map_inited     = false;
};

class FishToCylProj {
public:
    explicit FishToCylProj(const Undistorter& undistorter);
    void stitch_project_to_cyn();
    void setImage(cv::Mat input_img);
    void setExtraImage(cv::Mat extra_img);
    cv::Mat getProjectedImage();

private:
    uchar3* extra_view_buffer = nullptr;

    ViewGPU_stilib extra_view;
    ViewGPU_stilib view_;
    int cyl_image_width_, cyl_image_height_;
    int row_grid_num_ = 32, col_grid_num_ = 32;

    std::mutex mutex_on_back_track_image_buffer;
    cv::Mat back_track_image_buffer;

    std::shared_ptr<eCAL::protobuf::CSubscriber<proto_messages::OpencvImage>>
        m_sub_back_track = nullptr;

    // 图像回调函数
    void ecalBackTrackImageCallBack(const char* _topic_name,
                                    const proto_messages::OpencvImage& msg,
                                    long long int _time, long long int _clock,
                                    long long int _id) {
        cv::Mat image(msg.rows(), msg.cols(), msg.elt_type());

        image.data = (uchar*)(msg.mat_data().data());

        memcpy(image.data, (uchar*)msg.mat_data().data(),
               image.rows * image.cols * image.elemSize());

        std::unique_lock<std::mutex> locker(mutex_on_back_track_image_buffer);
        back_track_image_buffer = image.clone();
        locker.unlock();
    }

    // 获取当前图像
    cv::Mat getBackTrackImage() {
        std::unique_lock<std::mutex> locker(mutex_on_back_track_image_buffer);
        cv::Mat result = back_track_image_buffer.clone();
        locker.unlock();
        return result;
    }
};
#endif   // CALIB_GROUND_UNDISTORTER_H
