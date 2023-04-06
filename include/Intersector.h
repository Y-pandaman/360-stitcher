//
// Created by touch on 22-12-27.
//

#ifndef INC_360CODE_INTERSECTOR_H
#define INC_360CODE_INTERSECTOR_H

#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

class Intersector {
public:
    static bool intersectLineSegments(const float3& line_seg1_begin, const float3& line_seg1_end,
                                      const float3& line_seg2_begin, const float3& line_seg2_end){
        //快速排斥实验
//        if ((l1.x1 > l1.x2 ? l1.x1 : l1.x2) < (l2.x1 < l2.x2 ? l2.x1 : l2.x2) ||
//            (l1.y1 > l1.y2 ? l1.y1 : l1.y2) < (l2.y1 < l2.y2 ? l2.y1 : l2.y2) ||
//            (l2.x1 > l2.x2 ? l2.x1 : l2.x2) < (l1.x1 < l1.x2 ? l1.x1 : l1.x2) ||
//            (l2.y1 > l2.y2 ? l2.y1 : l2.y2) < (l1.y1 < l1.y2 ? l1.y1 : l1.y2))
//        {
//            return false;
//        }
        //跨立实验

        // AB
        Eigen::Vector3f seg1_vec(line_seg1_end.x - line_seg1_begin.x,
                                 line_seg1_end.y - line_seg1_begin.y,
                                 0);
        // AC AD
        Eigen::Vector3f AC_vec(line_seg2_begin.x - line_seg1_begin.x,
                               line_seg2_begin.y - line_seg1_begin.y,
                               0);
        Eigen::Vector3f AD_vec(line_seg2_end.x - line_seg1_begin.x,
                               line_seg2_end.y - line_seg1_begin.y,
                               0);
        float z0 = seg1_vec.cross(AC_vec).z();
        float z1 = seg1_vec.cross(AD_vec).z();
        if (z0 * z1 > 0) return false;
        // CD
        Eigen::Vector3f seg2_vec(line_seg2_end.x - line_seg2_begin.x,
                                 line_seg2_end.y - line_seg2_begin.y,
                                 0);
        // CA CB
        Eigen::Vector3f CA_vec(line_seg1_begin.x - line_seg2_begin.x,
                               line_seg1_begin.y - line_seg2_begin.y,
                               0);
        Eigen::Vector3f CB_vec(line_seg1_end.x - line_seg2_begin.x,
                               line_seg1_end.y - line_seg2_begin.y,
                               0);
        float z2 = seg2_vec.cross(CA_vec).z();
        float z3 = seg2_vec.cross(CB_vec).z();
        if(z2 * z3 > 0) return false;

//        if ((((l1.x1 - l2.x1)*(l2.y2 - l2.y1) - (l1.y1 - l2.y1)*(l2.x2 - l2.x1))*
//             ((l1.x2 - l2.x1)*(l2.y2 - l2.y1) - (l1.y2 - l2.y1)*(l2.x2 - l2.x1))) > 0 ||
//            (((l2.x1 - l1.x1)*(l1.y2 - l1.y1) - (l2.y1 - l1.y1)*(l1.x2 - l1.x1))*
//             ((l2.x2 - l1.x1)*(l1.y2 - l1.y1) - (l2.y2 - l1.y1)*(l1.x2 - l1.x1))) > 0)
//        {
//            return false;
//        }
        return true;
    }

    static bool intersectRectWithPointsOut(const float3& begin, const float3& end, const cv::Rect &rect){
        float x_min = rect.x, y_min = rect.y;
        float x_max = rect.x + rect.width, y_max = rect.height + rect.y;
        if (intersectLineSegments(begin, end,
                              make_float3(x_min, y_min, 0), make_float3(x_max, y_max, 0))) {
            return true;
        }
        if (intersectLineSegments(begin, end,
                                  make_float3(x_min, y_max, 0), make_float3(x_max, y_min, 0))){
            return true;
        }
        return false;
    }

    static bool pointInRect(const float3 &p, const cv::Rect &rect){
        if(p.x >= rect.x && p.x <= rect.x + rect.width && p.y >= rect.y && p.y <= rect.y + rect.height){
            return true;
        }
        else return false;
    }
    static bool intersectRects(const float3 ps[4], const cv::Rect &rect){
        for(int i = 0;i < 4;i++){
            if (pointInRect(ps[i], rect)) return true;
        }
        for(int i = 0;i < 4;i++){
            if(intersectRectWithPointsOut(ps[i], ps[(i + 1) % 4], rect)) return true;
        }
        return false;
    }

};


#endif //INC_360CODE_INTERSECTOR_H
