#pragma once

#if defined(_WIN32)
#pragma execution_character_set("utf-8")
#endif

#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <thrust/extrema.h>
#include "math_utils.h"

#include "helper_cuda.h"

// ============================================= XCHAO ===================================

struct PinholeCameraGPU {
    float fx, fy, cx, cy;
    float *R;
    float *T;
    float *C;

    PinholeCameraGPU() {}

    PinholeCameraGPU(float fx_, float fy_, float cx_, float cy_,
                     float *R_, float *T_, float *C_) {
        fx = fx_;
        fy = fy_;
        cx = cx_;
        cy = cy_;
        R = R_;
        T = T_;
        C = C_;
    }


    void showPara() {
        float *R_, *T_, *C_;
        cudaHostAlloc((void **) &R_, sizeof(float) * 9, cudaHostAllocDefault);
        cudaHostAlloc((void **) &T_, sizeof(float) * 3, cudaHostAllocDefault);
        cudaHostAlloc((void **) &C_, sizeof(float) * 3, cudaHostAllocDefault);

        cudaMemcpy(R_, R, sizeof(float) * 9, cudaMemcpyDeviceToHost);
        cudaMemcpy(T_, T, sizeof(float) * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_, C, sizeof(float) * 3, cudaMemcpyDeviceToHost);

        printf("fx: %f  fy: %f Cx: %f Cy: %f\n", fx, fy, cx, cy);
        printf("Rotation:\n");
        for (int i = 0; i < 3; i++) {
            printf("%f %f %f\n", R_[3 * i], R_[3 * i + 1], R_[3 * i + 2]);
        }
        printf("Translation Vector:\n");
        printf("%f %f %f\n", T_[0], T_[1], T_[2]);

        printf("Camera Center:\n");
        printf("%f %f %f\n", C_[0], C_[1], C_[2]);

        cudaFreeHost(R_);
        cudaFreeHost(T_);
        cudaFreeHost(C_);
    }

    inline __device__
            float3
    rotateVector(float3
    v)
    {
        return make_float3(
                R[0] * v.x + R[1] * v.y + R[2] * v.z,
                R[3] * v.x + R[4] * v.y + R[5] * v.z,
                R[6] * v.x + R[7] * v.y + R[8] * v.z);
    }

    inline __device__
            float3
    rotateVector_inv(float3
    v)
    {
        return make_float3(
                R[0] * v.x + R[3] * v.y + R[6] * v.z,
                R[1] * v.x + R[4] * v.y + R[7] * v.z,
                R[2] * v.x + R[5] * v.y + R[8] * v.z);
    }

    inline __device__
            float3

    getRay(int u, int v) {

        float3 dir = make_float3((u - cx) / fx, (v - cy) / fy, 1.0f);
        dir = this->rotateVector_inv(dir);
        return normalize(dir);
        // printf("dir %f %f %f", dir.x, dir.y, dir.z);
        // return dir;
    }

    inline __device__
            float3

    getCenter() {
        return make_float3(C[0], C[1], C[2]);
    }

    inline __device__
            float2
    projWorldToPixel(float3
    x)
    {
        x.x -= C[0];
        x.y -= C[1];
        x.z -= C[2];
        float3 cam = this->rotateVector(x);

        return make_float2(
                cam.x * fx / cam.z + cx,
                cam.y * fy / cam.z + cy);
    }
};

struct ViewGPU {
    uchar3 *image;
    uchar *mask;
    int height, width;
    PinholeCameraGPU camera;

    ViewGPU() {}

    ViewGPU(uchar3 *image_, uchar *mask_, int height_, int width_) {
        image = image_;
        mask = mask_;
        height = height_;
        width = width_;
    }

    ViewGPU(uchar3 *image_, uchar *mask_, int height_, int width_, PinholeCameraGPU camera_) {
        image = image_;
        mask = mask_;
        height = height_;
        width = width_;
        camera = camera_;
    }
};


struct CylinderGPU {
    float *rotation;
    float *center;
    float r;
    float *global_theta, *global_phi;
    int2 *boundary_pixel;
    // int size1, size2, size3;
    int offset[4] = {0, 0, 0, 0};

    CylinderGPU() {}

    CylinderGPU(float *rotation_, float *center_, float r_) {
        rotation = rotation_;
        center = center_;
        r = r_;
        cudaMalloc((void **) &global_theta, sizeof(float) * 2);
        cudaMalloc((void **) &global_phi, sizeof(float) * 2);
    }

    void setBoundary(std::vector<std::vector<int2>> boundary_loc) {
        offset[1] = boundary_loc[0].size();
        offset[2] = boundary_loc[0].size() + boundary_loc[1].size();
        offset[3] = boundary_loc[0].size() + boundary_loc[1].size() + boundary_loc[2].size();


        cudaMalloc((void **) &boundary_pixel, sizeof(int2) * offset[3]);

        cudaMemcpy(boundary_pixel + offset[0], boundary_loc[0].data(), sizeof(int2) * boundary_loc[0].size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(boundary_pixel + offset[1], boundary_loc[1].data(), sizeof(int2) * boundary_loc[1].size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(boundary_pixel + offset[2], boundary_loc[2].data(), sizeof(int2) * boundary_loc[2].size(),
                   cudaMemcpyHostToDevice);
    }

    inline __device__
            float3

    getCenter() {
        return make_float3(center[0], center[1], center[2]);
    }

    inline __device__
            float3
    rotateVector(float3
    v)
    {
        return make_float3(
                rotation[0] * v.x + rotation[1] * v.y + rotation[2] * v.z,
                rotation[3] * v.x + rotation[4] * v.y + rotation[5] * v.z,
                rotation[6] * v.x + rotation[7] * v.y + rotation[8] * v.z);
    }

    inline __device__
            float3
    rotateVector_inv(float3
    v)
    {
        return make_float3(
                rotation[0] * v.x + rotation[3] * v.y + rotation[6] * v.z,
                rotation[1] * v.x + rotation[4] * v.y + rotation[7] * v.z,
                rotation[2] * v.x + rotation[5] * v.y + rotation[8] * v.z);
    }
};


struct CylinderImageGPU {
    uchar3 *image;
    uchar *mask;
    int *uv;
    int height, width;

    CylinderImageGPU() {}

    CylinderImageGPU(uchar3 *image_, uchar *mask_, int height_, int width_) {
        image = image_;
        mask = mask_;
        height = height_;
        width = width_;

        cudaMalloc((void **) &uv, sizeof(int) * 2);
        cudaMemset(uv, 0, sizeof(int) * 2);
    }

    CylinderImageGPU(uchar3 *image_, uchar *mask_, int *uv_, int height_, int width_) {
        image = image_;
        mask = mask_;
        uv = uv_;
        height = height_;
        width = width_;
    }

    CylinderImageGPU(int height_, int width_) {
        height = height_;
        width = width_;
        cudaMalloc((void **) &image, sizeof(uchar3) * height * width);
        cudaMalloc((void **) &mask, sizeof(uchar) * height * width);
        cudaMalloc((void **) &uv, sizeof(int) * 2);
        cudaMemset(uv, 0, sizeof(int) * 2);

        // cudaError_t err = cudaGetLastError();
        // if ( err != cudaSuccess )
        // {
        //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
        //     printf("[ERROR] Create CylinderImageGPU: image&mask failed !!! \n");
        // }else{
        //   printf("Create CylinderImageGPU: image&mask\n");
        // }
    }

    void clear() {
        cudaFree(image);
        cudaFree(mask);
        cudaFree(uv);
    }

    void reAllocateMem(int height_, int width_) {
        height = height_;
        width = width_;
        cudaFree(image);
        cudaFree(mask);
        cudaMalloc((void **) &image, sizeof(uchar3) * height * width);
        cudaMalloc((void **) &mask, sizeof(uchar) * height * width);
        printf("Create CylinderImageGPU: image&mask\n");
    }

    bool toCPU(cv::Mat &image_, cv::Mat &mask_) {
#if 0
//        uchar3 *data_rgb;
//        uchar *data_mask;
//        int* data_uv;
//        cudaHostAlloc((void**)&data_rgb,sizeof(uchar3)*height*width,cudaHostAllocDefault);
//        cudaHostAlloc((void**)&data_mask,sizeof(uchar)*height*width,cudaHostAllocDefault);
//        cudaHostAlloc((void**)&data_uv,sizeof(int)*2,cudaHostAllocDefault);
//
//        cudaMemcpy(data_rgb, image, sizeof(uchar3)*height*width, cudaMemcpyDeviceToHost);
//        cudaMemcpy(data_mask, mask, sizeof(uchar)*height*width, cudaMemcpyDeviceToHost);
//        cudaMemcpy(data_uv, uv, sizeof(int)*2, cudaMemcpyDeviceToHost);
//
//        image_ = cv::Mat(height, width, CV_8UC3, data_rgb);
//        mask_ = cv::Mat(height, width, CV_8UC1, data_mask);
#endif
#if 1
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        uchar3 *data_rgb;
        uchar *data_mask;
//        int *data_uv;
        printf("toCPU() height: %d, width: %d\n", height, width);

        checkCudaErrors(cudaHostAlloc((void **) &data_rgb, sizeof(uchar3) * height * width, cudaHostAllocDefault));
        checkCudaErrors(cudaHostAlloc((void **) &data_mask, sizeof(uchar) * height * width, cudaHostAllocDefault));
//        checkCudaErrors(cudaHostAlloc((void **) &data_uv, sizeof(int) * 2, cudaHostAllocDefault));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(data_rgb, image, sizeof(uchar3) * height * width, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(data_mask, mask, sizeof(uchar) * height * width, cudaMemcpyDeviceToHost));
//        checkCudaErrors(cudaMemcpy(data_uv, uv, sizeof(int) * 2, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        image_ = cv::Mat(height, width, CV_8UC3, data_rgb);
        mask_ = cv::Mat(height, width, CV_8UC1, data_mask);

        image_ = image_.clone();
        mask_ = mask_.clone();
        cudaFreeHost(data_rgb);
        cudaFreeHost(data_mask);
        // std::cout << "uv x: " << data_uv[0] << " y: " << data_uv[1] << std::endl;
#endif
        return true;
    }

    inline __device__
            uchar

    getMaskValue(int x, int y) {
        // 输入拼接大图的坐标，通过corner找到对应的小图的坐标
        x = x - uv[0];
        y = y - uv[1];
        if (x < 0 || y < 0 || x >= width || y >= height)
            return 0;
        return mask[y * width + x];
    }

    inline __device__
            uchar3

    getImageValue(int x, int y) {
        x = x - uv[0];
        y = y - uv[1];
        if (x < 0 || y < 0 || x >= width || y >= height)
            return make_uchar3(0, 0, 0);
        return image[y * width + x];
    }
};


struct SeamImageGPU {
    short3 *image;
    short *mask;
    int height, width;

    SeamImageGPU() {}


    SeamImageGPU(short3 *image_, short *mask_, int height_, int width_) {
        image = image_;
        mask = mask_;
        height = height_;
        width = width_;
    }

    void clear() {
        cudaFree(image);
        cudaFree(mask);
    }
};

// ============================================= XCHAO ===================================


struct Plane {
    Eigen::Matrix<float, 3, 1> N;
    float d;

    Plane(Eigen::Matrix<float, 3, 1> _N, float _d) : N(_N), d(_d) {}
};


struct Cylinder {
    Eigen::Matrix<float, 3, 3> rotation;
    Eigen::Matrix<float, 3, 1> center;
    float r;

    Cylinder() : center(Eigen::Vector3f(0, 0, 0)), r(0.0f), rotation(Eigen::Matrix3f::Identity()) {}

    Cylinder(Eigen::Matrix<float, 3, 1> center, float r, Eigen::Matrix<float, 3, 3> rotation) : center(center), r(r),
                                                                                                rotation(rotation) {}

    CylinderGPU toGPU() {
        float *_rotation, *_center;
        cudaMalloc((void **) &_rotation, sizeof(float) * 9);
        cudaMalloc((void **) &_center, sizeof(float) * 3);

        // std::cout << "PROBLEM: COL MAJOR in Eigen Matrix (FIX ME) " << std::endl;
        Eigen::Matrix3f rotation_rowMajor = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(&rotation(0, 0));
        cudaMemcpy(_rotation, &rotation_rowMajor(0, 0), sizeof(float) * 9, cudaMemcpyHostToDevice);
        cudaMemcpy(_center, &center(0, 0), sizeof(float) * 3, cudaMemcpyHostToDevice);

        return CylinderGPU(_rotation, _center, r);
    }

};


struct Ray {
    Eigen::Matrix<float, 3, 1> origin;
    Eigen::Matrix<float, 3, 1> dir;

    Ray(Eigen::Matrix<float, 3, 1> origin, Eigen::Matrix<float, 3, 1> dir) : origin(origin), dir(dir) {
        this->dir.normalize();
    }

    Eigen::Matrix<float, 3, 1> getPoint(float t) {
        return this->origin + t * dir;
    }
};

static std::ostream &operator<<(std::ostream &out, const Ray &A) {
    out << "Ray o: " << A.origin.transpose() << "\tRay dir: " << A.dir.transpose() << "\n";
    return out;
}

struct PinholeCamera {
    Eigen::Matrix3f K;
    Eigen::Matrix3f R;
    Eigen::Matrix<float, 3, 1> T;
    Eigen::Matrix<float, 3, 1> C;

    PinholeCamera() {}

    PinholeCamera(Eigen::Matrix3f K, Eigen::Matrix3f R, Eigen::Matrix<float, 3, 1> T, bool c2w = false) : K(K) {
        if (c2w == false) {
            this->R = R;
            this->T = T;
            this->C = -1.0f * R.transpose() * T;
        } else {
            this->R = R.transpose();
            this->C = T;
            this->T = -1.0f * this->R * this->C;
        }
    }

    void setCenter(Eigen::Matrix<float, 3, 1> C_) {
        C = C_;
        T = -R * C;
    }

    Ray getRay(int u, int v) {
        // input pixel  return ray
        Eigen::Matrix<float, 3, 1> pixel;
        pixel << (float) u, (float) v, 1;
        Eigen::Matrix<float, 3, 1> dir = R.transpose() * K.inverse() * pixel;
        dir.normalize();
        return Ray(this->C, dir);
    }

    Eigen::Matrix<float, 2, 1> projToPixel(Eigen::Matrix<float, 3, 1> p) {
        Eigen::Matrix<float, 3, 1> m = this->K * this->R * (p - this->C);
        if (m(2, 0) == 0) {
            std::cout << "Error Proj" << std::endl;
        }
        float x = m(0, 0) / m(2, 0);
        float y = m(1, 0) / m(2, 0);

        Eigen::Matrix<float, 2, 1> out;
        out << x, y;
        return out;
    }

    PinholeCameraGPU toGPU() {
        float *_R, *_T, *_C;
        cudaMalloc((void **) &_R, sizeof(float) * 9);
        cudaMalloc((void **) &_T, sizeof(float) * 3);
        cudaMalloc((void **) &_C, sizeof(float) * 3);

        // PROBLRM: COL MAJOR
        // std::cout << "PROBLEM: COL MAJOR in Eigen Matrix (FIX ME) " << std::endl;
        Eigen::Matrix3f R_rowMajor = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(&R(0, 0));
        cudaMemcpy(_R, &R_rowMajor(0, 0), sizeof(float) * 9, cudaMemcpyHostToDevice);
        cudaMemcpy(_T, &T(0, 0), sizeof(float) * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(_C, &C(0, 0), sizeof(float) * 3, cudaMemcpyHostToDevice);

        return PinholeCameraGPU(K(0, 0), K(1, 1), K(0, 2), K(1, 2), _R, _T, _C);
    }
};

struct View {
    cv::Mat image;
    cv::Mat mask;
    PinholeCamera camera;

    View() {}

    View(cv::Mat image, cv::Mat mask, PinholeCamera camera) : image(image), mask(mask), camera(camera) {}


    ViewGPU toGPU() {
        PinholeCameraGPU cam = camera.toGPU();

        uchar3 *_image;
        uchar *_mask;
        cudaMalloc((void **) &_image, sizeof(uchar3) * image.rows * image.cols);
        cudaMalloc((void **) &_mask, sizeof(uchar) * mask.rows * mask.cols);
        cudaMemcpy(_image, image.ptr<uchar3>(0), sizeof(uchar3) * image.rows * image.cols, cudaMemcpyHostToDevice);
        cudaMemcpy(_mask, mask.ptr<uchar>(0), sizeof(uchar) * mask.rows * mask.cols, cudaMemcpyHostToDevice);


        return ViewGPU(_image, _mask, image.rows, image.cols, cam);
    }


};






// class CylinderStitcher {
// public:
//   Cylinder calCylinder(std::vector<PinholeCamera> cameras) {}
//   std::vector<CylinderImage> projToCylinderImage(std::vector<View> views, Cylinder cylinder);
//   std::vector<CylinderImage> alignImages(std::vector<CylinderImage> cylinder_images);
//   std::vector<cv::Mat> projToScreen(std::vector<CylinderImage> cylinder_images, std::vector<PinholeCamera> cameras) {}
//   void show(std::vector<cv::Mat> images) {}
//   void stitch(std::vector<View> views);

// private:
//   std::vector<View> views_;
//   std::vector<CylinderImage> cylinder_images_;
//   Cylinder cylinder_;
// };
