#include "Intersector.h"
#include "airview_stitcher.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "yolo_detect.h"
#include <filesystem>
#include <iostream>

class Quadrilateral {
public:
    Quadrilateral(std::vector<cv::Point> point_vec) : point(4) {
        for (int i = 0; i < point_vec.size(); i++)
            point[i] = point_vec[i];
    }

    void draw_on(cv::Mat image) {
        for (int i = 0; i < 4; i++)
            cv::line(image, point[i], point[(i + 1) % 4],
                     cv::Scalar(255, 255, 255), 4);
    }

    float3* convertToFloat3() {
        float3* result = new float3[4];
        for (int i = 0; i < 4; i++) {
            result[i] = make_float3(point[i].x, point[i].y, 0);
        }
        return result;
    }

    void move(int dx, int dy) {
        for (int i = 0; i < 4; i++) {
            point[i].x += dx;
            point[i].y += dy;
        }
    }

private:
    std::vector<cv::Point> point;
};
int main() {
    bool adjust_rect = false;
    std::filesystem::path parameters_dir(
        "/home/xcmg/stitcher/360-stitcher/parameters");
    std::filesystem::path weight_file_path(parameters_dir);
    weight_file_path.append("weights/best.onnx");
    std::cout << weight_file_path << std::endl;
    //    yolo_detect::YoloDetect yolo_detect(weight_file_path, true);
    checkCudaErrors(cudaFree(0));
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float milliseconds = 0;

    int src_height = 1296;
    int src_width  = 2304;
    int HEIGHT     = 2160;
    int WIDTH      = 3840;
    //    std::filesystem::path
    //    yamls_dir("/home/touch/data/360/parameters/yamls");
    std::filesystem::path yamls_dir("/home/xcmg/stitcher/data/922");
    //    yamls_dir.append("yamls");
    std::filesystem::path car_rect_yaml(yamls_dir);
    car_rect_yaml.append("car_rect.yaml");
    cv::FileStorage car_rect_fs;
    cv::Rect_<int> car_rect;
    if (!car_rect_fs.open(car_rect_yaml, cv::FileStorage::READ)) {
        car_rect.x      = WIDTH / 2 - 1000;
        car_rect.y      = HEIGHT / 2 - 500;
        car_rect.width  = 2000;
        car_rect.height = 1000;
    } else {
        //        car_rect_fs["x"] >> car_rect.x;
        //        car_rect_fs["y"] >> car_rect.y;
        //        car_rect_fs["width"] >> car_rect.width;
        //        car_rect_fs["height"] >> car_rect.height;
        car_rect_fs["car_rect"] >> car_rect;
    }
    car_rect_fs.release();

    // std::vector<int> camera_idx_vec = {0, 7, 2, 1, 3, 6};
    std::vector<int> camera_idx_vec = {1, 2, 4, 5, 3, 0};
    //    for(int i = 0 ;i < camera_idx_vec.size();i++) {
    //        cv::imshow(std::to_string(camera_idx_vec[i]),
    //        undistorted_image_vec[i]);
    //    }
    //    cv::waitKey(0);
    //    cv::Size undistorted_image_size =
    //    undistorter_vec[0].getNewImageSize(); int src_height =
    //    undistorted_image_size.height, src_width =
    //    undistorted_image_size.width;

    AirViewStitcher* As = new AirViewStitcher(camera_idx_vec.size(), src_height,
                                              src_width, HEIGHT, WIDTH, 2);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 读取 mask 和 homo
    std::vector<cv::Mat> masks(camera_idx_vec.size());
    std::vector<cv::Mat> Hs(camera_idx_vec.size());
    for (int i = 0; i < camera_idx_vec.size(); ++i) {
        cv::Mat mask_image;
        std::filesystem::path mask_path(yamls_dir);
        mask_path.append(std::to_string(camera_idx_vec[i]) + "_mask.png");
        mask_image = cv::imread(mask_path.c_str(), 0);
        //        undistorter_vec[i].getMask(mask_image);
        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);   // Creating distortion matrix
        std::filesystem::path yaml_path(yamls_dir);
        yaml_path.append("camera_" + std::to_string(camera_idx_vec[i]) +
                         "_homography.yaml");
        cv::FileStorage fs;
        fs.open(yaml_path, cv::FileStorage::READ);
        fs["H"] >> H;
        fs.release();
        H.convertTo(H, CV_32F);

        if (mask_image.rows != src_height) {
            float scale = 1.0f * src_width / mask_image.cols;
            std::cout << "mask size " + std::to_string(camera_idx_vec[i]) + ": "
                      << mask_image.size() << std::endl;
            cv::resize(mask_image, mask_image, cv::Size(src_width, src_height));
            std::cout << "resize " << camera_idx_vec[i] << " "
                      << " scale " << scale << " " << camera_idx_vec[i]
                      << std::endl;
            H.at<float>(0, 0) /= scale;
            H.at<float>(0, 1) /= scale;
            H.at<float>(1, 0) /= scale;
            H.at<float>(1, 1) /= scale;
            H.at<float>(2, 0) /= scale;
            H.at<float>(2, 1) /= scale;
        }
        masks[i] = mask_image;
        Hs[i]    = H;
    }
    //    for(int i = 0; i < masks.size();i++){
    //        cv::imshow("mask " + std::to_string(camera_idx_vec[i]), masks[i]);
    //    }
    //    cv::waitKey(0);
    As->init(masks, Hs);
    // As->setIcon("car.png", 0.45f);

    std::vector<std::vector<BBox>> bboxs(camera_idx_vec.size());
    for (int i = 0; i < bboxs.size(); i++) {
        std::vector<BBox> temp;
        bboxs[i] = temp;
    }

    int frame_count = 0;

    std::vector<cv::Mat> undistorted_image_vec;
    std::filesystem::path image_dir(
        "/home/xcmg/stitcher/data/922/camera_calib");

    for (int i = 0; i < camera_idx_vec.size(); i++) {
        cv::Mat img;
        std::filesystem::path image_path(image_dir);
        image_path.append(std::to_string(camera_idx_vec[i]) + "-0.png");
        img = cv::imread(image_path);
        undistorted_image_vec.push_back(img);
        if (undistorted_image_vec[i].rows != src_height) {
            //            printf("%d need resize\n", camera_idx_vec[i]);
            cv::resize(undistorted_image_vec[i], undistorted_image_vec[i],
                       cv::Size(src_width, src_height));
            //            printf("%d resize done\n", camera_idx_vec[i]);
        }
    }

    As->feed(undistorted_image_vec, bboxs);
    cv::Mat rgb = As->output_CPUMat();
    //    cv::imshow("rgb", rgb);
    //    cv::waitKey(0);
    delete As;

    cv::VideoWriter video_writer;
    video_writer.open("./video/video_intersect.avi",
                      cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20,
                      cv::Size(WIDTH, HEIGHT), true);
    cv::Scalar warn_color(0, 0, 255), safe_color(255, 0, 0);
    cv::Point step(3, 0);
    std::vector<cv::Point> point_vec(4);
    point_vec[0] = cv::Point(800 - 400, 400);
    point_vec[1] = cv::Point(1300 - 400, 400);
    point_vec[2] = cv::Point(1200 - 400, 700);
    point_vec[3] = cv::Point(800 - 400, 700);
    Quadrilateral quadrilateral(point_vec);
    int max_frame_count = 500;
    frame_count         = 0;
    srand((int)time(0));
    int max_rand = 4;
    while (frame_count < max_frame_count) {
        //        cv::Point cur_step(step);
        //        cur_step.y += rand()%max_rand - max_rand / 2;
        quadrilateral.move(step.x, step.y - frame_count / 150);
        cv::Scalar car_rect_color = safe_color;
        cv::Mat rgb_clone         = rgb.clone();
        // test intersector::intersectRect
        quadrilateral.draw_on(rgb_clone);
        if (Intersector::intersectRects(quadrilateral.convertToFloat3(),
                                        car_rect)) {
            //            intersect_warning = true;
            car_rect_color = warn_color;
        }
        cv::rectangle(rgb_clone, car_rect, car_rect_color, 4);
        //        cv::imwrite("video/rgb_clone.png", rgb_clone);
        video_writer.write(rgb_clone);
        printf("frame = %d\n", frame_count);
        frame_count++;
        //        break;
    }
    return 0;
}