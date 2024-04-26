//
// Created by touch on 22-12-27.
//

#ifndef INC_360CODE_INTERSECTOR_H
#define INC_360CODE_INTERSECTOR_H

#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>

class Intersector {
public:
    static bool pointInRect(const float3& p, const cv::Rect& rect) {
        if (p.x >= rect.x && p.x <= rect.x + rect.width && p.y >= rect.y &&
            p.y <= rect.y + rect.height) {
            return true;
        } else
            return false;
    }
    static bool intersectRects(const float3 ps[4], const cv::Rect& rect) {
        for (int i = 0; i < 4; i++) {
            if (pointInRect(ps[i], rect))
                return true;
        }
        return false;
    }
};

#endif   // INC_360CODE_INTERSECTOR_H
