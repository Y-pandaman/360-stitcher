#include "GstReceiver.h"

/**
 * 初始化函数，用于打开视频流并设置图像队列。
 *
 * @param url_ 视频流的URL地址。
 * @param queue_size_ 图像队列的大小，用于缓存视频帧。
 * @return 如果初始化成功返回true，否则返回false。
 */
bool GstReceiver::initialize(const std::string& url_, int queue_size_) {
    this->video_url  = url_;
    this->queue_size = queue_size_;

    // 如果对象已经初始化过，则重新初始化图像队列和互斥锁
    if (status != UNINITIALIZED) {
        delete[] this->image_queue;
        delete[] this->mutex_on_image_queue;
    }
    // 创建新的图像队列和对应的互斥锁
    this->image_queue          = new cv::Mat[queue_size_];
    this->mutex_on_image_queue = new std::mutex[queue_size_];

    // 尝试打开视频流
    video_capture.open(this->video_url);
    // 如果视频流无法打开，则返回false
    if (!video_capture.isOpened()) {
        printf("can't open %s \n", this->video_url.c_str());
        return false;
    }

    // 设置状态为已初始化
    status = INITIALIZED;
    return true;
}

/**
 * @brief 启动接收工作线程
 *
 * 该函数用于启动一个工作线程，用于抓取和处理视频流。在启动前，会检查当前状态，
 * 仅当状态为未初始化（UNINITIALIZED）时返回错误。如果状态为初始化（INITIALIZED）
 * 或停止（STOP），则会尝试重新打开视频源，并启动抓取视频的工作线程。
 *
 * @return bool - 如果成功启动工作线程，返回true；否则返回false。
 */
bool GstReceiver::startReceiveWorker() {
    if (status == UNINITIALIZED) {
        return false;   // 当状态为未初始化时，直接返回错误
    }
    if (status == INITIALIZED || status == STOP) {
        frame_count = 0;   // 重置帧计数
        // 检查视频源是否成功打开
        if (!video_capture.isOpened()) {
            printf("can't open %s \n", this->video_url.c_str());
            return false;   // 如果无法打开视频源，返回错误
        }
        // 获取视频高度和宽度
        video_height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        video_width  = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
        p_write = p_read = 0;   // 重置指针位置

        // 启动视频抓取工作线程
        thread_receive_worker =
            new std::thread(&GstReceiver::receiveWorker, this);
        status = RUN;   // 更新状态为运行中
    }
    return true;   // 成功启动工作线程，返回成功
}

/**
 * @brief 该函数是一个工作线程，用于从视频源持续读取图像并处理。
 * 它会不断尝试从视频源读取图像，将其放入一个队列中，直到停止标志被设置。
 *
 * @note
 * 该函数没有参数和返回值，因为它是一个成员函数，通过类的实例访问成员变量。
 */
void GstReceiver::receiveWorker() {
    while (true) {
        // 检查停止标志，如果被设置，则清理并退出循环
        if (stop_flag) {
            stop_flag = false;
            //            status = STOP;
            return;
        }

        // 计算要写入的队列位置
        int pos = p_write % queue_size;
        // 使用互斥锁锁定图像队列的位置，以确保线程安全
        std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);
        // 尝试从视频源读取图像
        bool read_flag = video_capture.read(image_queue[pos]);

        // 如果读取失败，则解锁并退出循环
        if (!read_flag) {
            locker.unlock();
            break;
        }

        // 克隆图像以供发布或其他处理
        cv::Mat image_to_pub = image_queue[pos].clone();
        // 更新写入位置
        p_write = (p_write + 1);
        // 解锁图像队列
        locker.unlock();
        // 通知其他等待图像的线程，队列中现在有图像了
        cond_image_queue_not_empty.notify_one();
    }
}

/**
 * @brief 获取图像数据
 *
 * 该函数从图像队列中获取数据，如果队列为空，则等待数据就绪。获取数据后，会创建一个新的uchar数组来存储数据，
 * 并返回这个数组的指针。调用者需要确保在使用完数据后释放这个返回的数组。
 *
 * @return uchar* 指向从图像队列中获取的数据的指针。调用者需要负责释放这块内存。
 */
uchar* GstReceiver::getImageData() {
    // 根据循环队列的逻辑，计算当前读取位置
    int pos = p_read % queue_size;
    // 加锁以保护图像队列的访问
    std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);
    // 如果读指针大于等于写指针，说明队列为空，等待队列非空的通知
    while (p_read >= p_write) {
        cond_image_queue_not_empty.wait(locker);
    }
    // 计算所需缓冲区的大小
    size_t buf_size = image_queue[pos].rows * image_queue[pos].cols *
                      image_queue[pos].elemSize();
    // 分配内存以存储图像数据
    uchar* result = new uchar[buf_size];
    // 将图像数据从队列复制到新分配的内存中
    memcpy(result, image_queue[pos].data, buf_size);
    // 更新读指针，准备读取下一个图像数据
    p_read = p_read + 1;
    // 解锁
    locker.unlock();
    // 返回图像数据的指针
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
}

/**
 * 从图像队列中获取一张Mat图像。
 * 此函数是线程安全的，会等待直到队列中有图像可用。
 *
 * @return cv::Mat 返回从队列中获取到的图像Mat对象。
 */
cv::Mat GstReceiver::getImageMat() {
    cv::Mat result;

    // 如果队列中的图像数量大于等于1，则准备读取最新的一帧图像
    if (p_write >= 1)
        p_read = p_write - 1;

    // 计算当前要读取的队列位置
    int pos = p_read % queue_size;

    // 加锁以保护图像队列，确保线程安全
    std::unique_lock<std::mutex> locker(mutex_on_image_queue[pos]);

    // 等待直到队列中有图像可供读取
    while (p_read >= p_write) {
        cond_image_queue_not_empty.wait(locker);
    }

    // 从队列中克隆出图像，并更新读取位置
    result = image_queue[pos].clone();
    p_read = (p_read + 1);

    // 解锁图像队列
    locker.unlock();

    return result;
}

// 设置去畸变矩阵
/**
 * 设置图像去畸变器的相关参数并初始化。
 *
 * @param intrin_file_path 内参文件路径，用于加载相机的内参。
 * @param new_size_factor 新图像尺寸因子，用于调整输出图像的大小。
 * @param balance 平衡因子，用于调整去畸变过程中图像质量与速度的平衡。
 * @param USE_720P 是否使用720P分辨率。如果为true，则将输出图像尺寸调整为720P。
 */
void GstReceiver::setUndistorter(const std::string& intrin_file_path,
                                 float new_size_factor, float balance,
                                 bool USE_720P) {
    // 加载相机内参文件
    undistorter.loadCameraIntrin(intrin_file_path);
    // 如果启用720P，则调整图像尺寸
    if (USE_720P) {
        undistorter.changeSize(2.0 / 3.0);
    }
    // 根据参数获取重映射用的映射图
    undistorter.getMapForRemapping(new_size_factor, balance);
}
