#include "GstReceiver.h"

// void
bool GstReceiver::initialize(const std::string& url_, int queue_size_) {
    this->video_url  = url_;
    this->queue_size = queue_size_;
    if (status != UNINITIALIZED) {
        delete[] this->image_queue;
        delete[] this->mutex_on_image_queue;
    }
    this->image_queue          = new cv::Mat[queue_size_];
    this->mutex_on_image_queue = new std::mutex[queue_size_];

    video_capture.open(this->video_url);
    if (!video_capture.isOpened()) {
        printf("can't open %s \n", this->video_url.c_str());
        return false;
    }

    status = INITIALIZED;
    return true;
}

bool GstReceiver::startReceiveWorker() {
    if (status == UNINITIALIZED) {
        return false;
    }
    if (status == INITIALIZED || status == STOP) {
        frame_count = 0;
        if (!video_capture.isOpened()) {
            printf("can't open %s \n", this->video_url.c_str());
            return false;
        }
        video_height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        video_width  = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
        p_write = p_read = 0;
        thread_receive_worker =
            new std::thread(&GstReceiver::receiveWorker, this);
        status = RUN;
    }
    return true;
}

void GstReceiver::receiveWorker() {
    while (true) {
        if (stop_flag) {
            stop_flag = false;
            //            status = STOP;
            return;
        }
        int pos             = p_write % queue_size;
        int cur_write_frame = p_write;
        std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);
        bool read_flag = video_capture.read(image_queue[pos]);

        if (!read_flag) {
            locker.unlock();
            break;
        }
        cv::Mat image_to_pub = image_queue[pos].clone();
        p_write              = (p_write + 1);
        locker.unlock();

        //        if(flag_undistort_image){
        //            undistorter.undistortImage(image_to_pub, image_to_pub);
        //        }
        //        if(flag_yolo_detect && yolo_detector != nullptr){
        //            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        //            LOG_F(INFO, "detect %d ...", cur_write_frame);
        //            image_to_pub = yolo_detector->detect(image_to_pub);
        //            LOG_F(INFO, "detect %d done", cur_write_frame);
        //        }
        //        if (flag_send_original_image) {
        //            ecal_original_image_sender.pubImage(image_to_pub);
        //        }
        cond_image_queue_not_empty.notify_one();
    }
}

uchar* GstReceiver::getImageData() {
    int pos = p_read % queue_size;
    std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);
    while (p_read >= p_write) {
        cond_image_queue_not_empty.wait(locker);
    }
    size_t buf_size = image_queue[pos].rows * image_queue[pos].cols *
                      image_queue[pos].elemSize();
    uchar* result = new uchar[buf_size];
    memcpy(result, image_queue[pos].data, buf_size);
    p_read = p_read + 1;
    locker.unlock();
    return result;
}

GstReceiver::~GstReceiver() {
    if (thread_receive_worker != nullptr) {
        thread_receive_worker->join();
    }
    if (thread_write_worker != nullptr) {
        thread_write_worker->join();
    }
    delete image_queue;
    delete mutex_on_image_queue;
    delete thread_receive_worker;
    delete thread_write_worker;
    delete yolo_detector;
}

cv::Mat GstReceiver::getImageMat() {
    cv::Mat result;
    if (p_write >= 1)
        p_read = p_write - 1;
    int pos = p_read % queue_size;
    std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);
    while (p_read >= p_write) {
        cond_image_queue_not_empty.wait(locker);
    }
    result = image_queue[pos].clone();
    p_read = (p_read + 1);
    locker.unlock();
    return result;
}

void GstReceiver::setUndistorter(const std::string& intrin_file_path,
                                 float new_size_factor, float balance,
                                 bool USE_720P) {
    undistorter.loadCameraIntrin(intrin_file_path);
    if (USE_720P) {
        undistorter.changeSize(2.0 / 3.0);
    }
    undistorter.getMapForRemapping(new_size_factor, balance);
    flag_undistort_image = true;
}

void GstReceiver::setYoloDetector(const std::string& weight_file_path) {
    if (yolo_detector != nullptr) {
        printf("already set yolo detector\n");
        return;
    }
    yolo_detector    = new yolo_detect::YoloDetect(weight_file_path, true);
    flag_yolo_detect = true;
}