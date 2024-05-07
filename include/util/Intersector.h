/*
 * @Author: 姚潘涛
 * @Date: 2024-04-25 20:53:06
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-07 08:54:12
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
//
// Created by touch on 22-12-27.
//

#ifndef INC_360CODE_INTERSECTOR_H
#define INC_360CODE_INTERSECTOR_H

#include <opencv2/opencv.hpp>

class Intersector {
public:
    /**
     * 判断三维点是否在二维矩形内
     *
     * @param p 三维点，需要是一个float3类型的变量，包含x、y、z坐标。
     * @param rect
     * 二维矩形，使用cv::Rect类型表示，包含左上角点的x、y坐标以及矩形的宽度和高度。
     * @return 返回一个布尔值，如果点在矩形内部则为true，否则为false。
     */
    static bool pointInRect(const float3& p, const cv::Rect& rect) {
        // 检查点的x坐标是否在矩形的x坐标范围内，并且点的y坐标是否在矩形的y坐标范围内
        if (p.x >= rect.x && p.x <= rect.x + rect.width && p.y >= rect.y &&
            p.y <= rect.y + rect.height) {
            return true;
        } else
            return false;
    }
    /**
     * 检查四个点是否与给定的矩形相交
     *
     * @param ps 包含四个点坐标的数组，每个点为 float3 类型（x, y, z）
     * @param rect OpenCV 标准矩形对象，表示一个矩形区域
     * @return 如果任意一个点在矩形内，则返回 true，否则返回 false
     */
    static bool intersectRects(const float3 ps[4], const cv::Rect& rect) {
        // 遍历四个点，检查每个点是否在矩形内
        for (int i = 0; i < 4; i++) {
            if (pointInRect(ps[i], rect))
                return true;   // 一旦发现有任一点在矩形内，立即返回 true
        }
        return false;   // 所有四个点都不在矩形内，返回 false
    }
};

#endif   // INC_360CODE_INTERSECTOR_H
