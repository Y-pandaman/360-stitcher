
#ifndef CYLINDER_STITCHER_GSTRECEIVER_H
#define CYLINDER_STITCHER_GSTRECEIVER_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <string>

#include "EcalImageSender.h"
#include "Undistorter.h"
#include "yolo_detect.h"
#include "loguru.hpp"
//#include "yolo.h"

class GstReceiver {
public:
    bool initialize(const std::string &url_, int queue_size_ = 2);
    bool startReceiveWorker();
//    bool startWriteWorker();
    uchar *getImageData();
    cv::Mat getImageMat();
    ~GstReceiver();

    void setUndistorter(const std::string &intrin_file_path, float new_size_factor = 1.0, float balance = 0.0,
                        bool USE_720P = false);
    void setYoloDetector(const std::string &weight_file_path);
//    void openTopicToSendOriginalImage(const std::string &topic);

private:
    int frame_count =0;
    void receiveWorker();
//    void writeWorker();
    std::condition_variable cond_image_queue_not_empty;
    std::atomic_bool stop_flag = false;
    cv::VideoCapture video_capture;
    int queue_size;
    int video_width, video_height;
    std::string video_url;
    cv::Mat *image_queue = nullptr;
    std::mutex *mutex_on_image_queue = nullptr;
    std::thread *thread_receive_worker = nullptr, *thread_write_worker = nullptr;
//    std::mutex mutex_on_out_buffer;
//    std::deque<cv::Mat> out_buffer;


    std::atomic_ullong p_write = 0, p_read = 0;
    enum Status{
        UNINITIALIZED, INITIALIZED, RUN, STOP
    } status = UNINITIALIZED;

//    EcalImageSender ecal_original_image_sender;
    Undistorter undistorter;
    std::atomic_bool flag_send_original_image = false, flag_undistort_image = false, flag_yolo_detect = false;
    yolo_detect::YoloDetect *yolo_detector = nullptr;
};


#endif //CYLINDER_STITCHER_GSTRECEIVER_H
