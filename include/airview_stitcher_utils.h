#ifndef __AIRVIEW_STITCHER__UTILS__
#define __AIRVIEW_STITCHER__UTILS__

#include "cylinder_stitcher.h"

#define MASK_MAX 255

//#define PRINT_WEIGHT_ON_SEAM

typedef std::pair<int, int2> myPair;

__device__ __constant__ static float GauKernel[25] = {
    0.0039, 0.0156, 0.0234, 0.0156, 0.0039, 0.0156, 0.0625, 0.0938, 0.0625,
    0.0156, 0.0234, 0.0938, 0.1406, 0.0938, 0.0234, 0.0156, 0.0625, 0.0938,
    0.0625, 0.0156, 0.0039, 0.0156, 0.0234, 0.0156, 0.0039};

const int2 search_seq[8] = {
    make_int2(1, 0),  make_int2(-1, 0), make_int2(0, -1), make_int2(0, 1),
    make_int2(-1, 1), make_int2(1, 1),  make_int2(1, -1), make_int2(-1, -1)};

struct cmp {
    template <typename T, typename U>
    bool operator()(T const& left, U const& right) {
        if (left.first > right.first)
            return true;
        return false;
    }
};

static inline float3 homographBased_warp(float3 p, cv::Mat H) {
    float3 warped_p;
    warped_p.x = H.at<float>(0, 0) * p.x + H.at<float>(0, 1) * p.y +
                 H.at<float>(0, 2) * p.z;
    warped_p.y = H.at<float>(1, 0) * p.x + H.at<float>(1, 1) * p.y +
                 H.at<float>(1, 2) * p.z;
    warped_p.z = H.at<float>(2, 0) * p.x + H.at<float>(2, 1) * p.y +
                 H.at<float>(2, 2) * p.z;

    return warped_p;
}

static inline bool is_boundary(cv::Mat src, int i, int j) {
    if (i == 0 || j == 0 || i == src.rows - 1 || j == src.cols - 1)
        return true;
    for (int k = 0; k < 8; k++) {
        if (src.at<uchar>(i + search_seq[k].x, j + search_seq[k].y) < 128)
            return true;
    }
    return false;
}

static inline int count_mask(cv::Mat mask, int thresh = 128) {
    int count = 0;
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i, j) > thresh)
                count++;
        }
    }
    return count;
}

static inline bool get_overlapMask(std::vector<cv::Mat>& masks,
                                   std::vector<cv::Mat>& overlap_masks) {
    for (int i = 0; i < masks.size(); i++) {
        int j               = (i + 1) % masks.size();
        int k               = (j + 1) % masks.size();
        cv::Mat tri_overlap = masks[i] & masks[j] & masks[k];
        int count           = count_mask(tri_overlap);
        if (count > 0) {
            printf("%d, %d, %d mask -> tri overlap\n", i, j, k);

            int count_i = count_mask(masks[i]);
            int count_j = count_mask(masks[j]);
            int count_k = count_mask(masks[k]);
            if (count_i > count_j) {
                if (count_i > count_k) {
                    masks[i] = masks[i] - tri_overlap;
                } else {
                    masks[k] = masks[k] - tri_overlap;
                }
            } else {
                if (count_j > count_k) {
                    masks[j] = masks[j] - tri_overlap;
                } else {
                    masks[k] = masks[k] - tri_overlap;
                }
            }
        }
    }
    return true;
}

static inline bool adjust_overlapmask(cv::Mat& a, cv::Mat& b) {
    assert(a.rows == b.rows && a.cols == b.cols);

    cv::Mat c       = a & b;
    bool is_overlap = false;
    for (int i = 0; i < c.rows && !is_overlap; i++) {
        for (int j = 0; j < c.cols; j++) {
            if (c.at<uchar>(i, j) > 128) {
                is_overlap = true;
                break;
            }
        }
    }

    if (!is_overlap)
        return is_overlap;

    int count_a = 0, count_b = 0;
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            if (a.at<uchar>(i, j) > 128)
                count_a++;
            if (b.at<uchar>(i, j) > 128)
                count_b++;
        }
    }

    if (count_a > count_b) {
        a = a - c;
    } else {
        b = b - c;
    }
    return is_overlap;
}

static inline void DFS_collect_overlap_point(int2 loc, int mark_number,
                                             cv::Mat marks,
                                             std::vector<int2>& overlap_points,
                                             cv::Mat& visited) {
    overlap_points.emplace_back(loc);

    visited.at<uchar>(loc.x, loc.y) = 1;

    for (int k = 0; k < 8; k++) {
        int2 next_loc = loc + search_seq[k];

        if (next_loc.x < 0 || next_loc.y < 0 || next_loc.x >= visited.rows ||
            next_loc.y >= visited.cols)
            continue;

        if (visited.at<uchar>(next_loc.x, next_loc.y) == 1)
            continue;

        if (marks.at<uchar>(next_loc.x, next_loc.y) != mark_number)
            continue;

        DFS_collect_overlap_point(next_loc, mark_number, marks, overlap_points,
                                  visited);
    }
}

static inline int4 decide_start_end(cv::Mat overlap_mask, cv::Mat mask0,
                                    cv::Mat mask1) {
    // STEP1 标记 oevrlap的边缘区域
    cv::Mat marks =
        cv::Mat::zeros(overlap_mask.rows, overlap_mask.cols, CV_8UC1);
    for (int i = 0; i < overlap_mask.rows; i++) {
        for (int j = 0; j < overlap_mask.cols; j++) {
            if (overlap_mask.at<uchar>(i, j) != 255)
                continue;

            if (is_boundary(overlap_mask, i, j) && is_boundary(mask0, i, j) &&
                is_boundary(mask1, i, j)) {
                marks.at<uchar>(i, j) = 255;
            }
        }
    }
    // STEP2 收集overlap边缘, 理论上应该有两条
    cv::Mat visited =
        cv::Mat::zeros(overlap_mask.rows, overlap_mask.cols, CV_8UC1);
    std::vector<std::vector<int2>> overlap_points;
    for (int i = 0; i < marks.rows; i++) {
        for (int j = 0; j < marks.cols; j++) {
            if (marks.at<uchar>(i, j) != 255)
                continue;
            if (visited.at<uchar>(i, j) == 1)
                continue;

            std::vector<int2> temp;
            DFS_collect_overlap_point(make_int2(i, j), 255, marks, temp,
                                      visited);
            overlap_points.emplace_back(temp);
        }
    }

    // std::cout << "overlap_points size " << overlap_points.size() << std::endl;
    assert(overlap_points.size() >= 2);

    int2 start = overlap_points[0][overlap_points[0].size() / 2];
    int2 end   = overlap_points[1][overlap_points[1].size() / 2];

    // 选择距离图像中心近的点为起点
    int2 center = make_int2(overlap_mask.rows / 2, overlap_mask.cols / 2);
    int d1      = (start.x - center.x) * (start.x - center.x) +
             (start.y - center.y) * (start.y - center.y);
    int d2 = (end.x - center.x) * (end.x - center.x) +
             (end.y - center.y) * (end.y - center.y);
    if (d1 > d2) {
        return make_int4(end.x, end.y, start.x, start.y);
    }
    return make_int4(start.x, start.y, end.x, end.y);
}

static inline void Dijkstra_search_seam(int2 start, int2 end, cv::Mat diff,
                                        cv::Mat overlap_mask, cv::Mat& seam_map,
                                        cv::Mat& seam_line, int seam_mark) {
    // 记录当前的顶点集合 CV_8UC1
    cv::Mat visited = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
    // 记录从起点到当前点的最短路径 float CV_32FC1
    cv::Mat D = cv::Mat::ones(diff.rows, diff.cols, CV_32FC1) * 100000000;
    // 记录每个pixel的prev点 类型 CV_32FC2
    cv::Mat path = cv::Mat::ones(diff.rows, diff.cols, CV_32FC2) * -1;

    D.at<float>(start.x, start.y) = 0;
    //    visited.at<uchar>(start.x, start.y) = 1;

    std::priority_queue<myPair, std::vector<myPair>, cmp> Q;
    Q.emplace(0, start);

    while (!Q.empty()) {
        int2 loc = Q.top().second;
        Q.pop();
        while (!Q.empty() && visited.at<uchar>(loc.x, loc.y) == 1) {
            loc = Q.top().second;
            Q.pop();
        }

        if (visited.at<uchar>(loc.x, loc.y) == 1) {
            break;   // all in visited
        }

        visited.at<uchar>(loc.x, loc.y) = 1;

        for (auto k = 0; k < 4; k++) {
            int2 next_loc = loc + search_seq[k];

            if (next_loc.x < 0 || next_loc.y < 0 || next_loc.x >= diff.rows ||
                next_loc.y >= diff.cols)
                continue;
            if (seam_map.at<uchar>(next_loc.x, next_loc.y) != 255)
                continue;
            if (overlap_mask.at<uchar>(next_loc.x, next_loc.y) == 0)
                continue;
            if (visited.at<uchar>(next_loc.x, next_loc.y) == 1)
                continue;
            float w =
                ((float)diff.at<ushort>(next_loc.x, next_loc.y) / 65535.0f);

            if (D.at<float>(next_loc.x, next_loc.y) >
                D.at<float>(loc.x, loc.y) + w) {
                D.at<float>(next_loc.x, next_loc.y) =
                    D.at<float>(loc.x, loc.y) + w;
                Q.emplace(D.at<float>(next_loc.x, next_loc.y), next_loc);

                path.at<float2>(next_loc.x, next_loc.y) =
                    make_float2(loc.x, loc.y);
            }
        }
    }

    // 确定终点
    float min_dis = 100000000.0f;
    int2 end_pts  = end;
    for (int i = 0; i < overlap_mask.rows; i++) {
        if (overlap_mask.at<uchar>(i, 0) != 0 && min_dis > D.at<float>(i, 0)) {
            min_dis = D.at<float>(i, 0);
            end_pts = make_int2(i, 0);
        }
        if (overlap_mask.at<uchar>(i, overlap_mask.cols - 1) != 0 &&
            min_dis > D.at<float>(i, overlap_mask.cols - 1)) {
            min_dis = D.at<float>(i, overlap_mask.cols - 1);
            end_pts = make_int2(i, overlap_mask.cols - 1);
        }
    }
    for (int i = 0; i < overlap_mask.cols; i++) {
        if (overlap_mask.at<uchar>(0, i) != 0 && min_dis > D.at<float>(0, i)) {
            min_dis = D.at<float>(0, i);
            end_pts = make_int2(0, i);
        }
        if (overlap_mask.at<uchar>(overlap_mask.rows - 1, i) != 0 &&
            min_dis > D.at<float>(overlap_mask.rows - 1, i)) {
            min_dis = D.at<float>(overlap_mask.rows - 1, i);
            end_pts = make_int2(overlap_mask.rows - 1, i);
        }
    }
    while (true) {
        seam_map.at<uchar>(end_pts.x, end_pts.y)  = seam_mark;
        seam_line.at<uchar>(end_pts.x, end_pts.y) = 255;

#ifdef PRINT_WEIGHT_ON_SEAM
        printf("%d ", diff.at<ushort>(end_pts.x, end_pts.y));
#endif
        if (end_pts.x == start.x && end_pts.y == start.y)
            break;

        float2 prev_loc = path.at<float2>(end_pts.x, end_pts.y);
        end_pts.x       = (int)prev_loc.x;
        end_pts.y       = (int)prev_loc.y;
    }
#ifdef PRINT_WEIGHT_ON_SEAM
    printf("\n");
#endif
}

static inline void seam_search(int4 endPts, cv::Mat overlap_mask, cv::Mat diff,
                               cv::Mat& seam_map, cv::Mat& seam_line,
                               int seam_mark) {
    int2 start = make_int2(endPts.x, endPts.y);
    int2 end   = make_int2(endPts.z, endPts.w);
    Dijkstra_search_seam(start, end, diff, overlap_mask, seam_map, seam_line,
                         seam_mark);
}

static inline void
BFS_get_seam_mask(int2 start,
                  std::vector<bool>& seam_flag,   //  标记当前区域相关的两条seam
                  cv::Mat total_mask, cv::Mat seam_map, cv::Mat& seam_mask,
                  cv::Mat& visited) {
    std::queue<int2> Q;

    Q.push(start);

    visited.at<uchar>(start.x, start.y)   = 1;
    seam_mask.at<uchar>(start.x, start.y) = 255;

    while (!Q.empty()) {
        int2 loc = Q.front();
        Q.pop();

        // 如果当前点是seam, 那么不可以再push点
        if (seam_map.at<uchar>(loc.x, loc.y) != 255) {
            continue;
        }

        for (int k = 0; k < 4; k++) {
            int2 next_loc = loc + search_seq[k];

            if (next_loc.x < 0 || next_loc.y < 0 ||
                next_loc.x >= visited.rows || next_loc.y >= visited.cols)
                continue;

            // 需要在total mask内部
            if (total_mask.at<uchar>(next_loc.x, next_loc.y) < 128)
                continue;

            if (seam_map.at<uchar>(next_loc.x, next_loc.y) != 255) {
                seam_flag[seam_map.at<uchar>(next_loc.x, next_loc.y)] = true;
            }

            if (visited.at<uchar>(next_loc.x, next_loc.y) == 1)
                continue;

            Q.push(next_loc);

            visited.at<uchar>(next_loc.x, next_loc.y)   = 1;
            seam_mask.at<uchar>(next_loc.x, next_loc.y) = 255;
        }
    }
}

static inline std::vector<cv::Mat>
gen_seam_mask(cv::Mat total_mask, cv::Mat total_seam_map, int num_view) {
    std::vector<cv::Mat> seam_masks(num_view);
    cv::Mat visited = cv::Mat::zeros(total_mask.rows, total_mask.cols, CV_8UC1);

    int mark_idx = 0;
    for (int i = 0; i < total_mask.rows; i++) {
        for (int j = 0; j < total_mask.cols; j++) {
            if (total_seam_map.at<uchar>(i, j) != 255)
                continue;
            if (visited.at<uchar>(i, j) == 1)
                continue;
            if (total_mask.at<uchar>(i, j) < 128)
                continue;

            std::vector<bool> seam_flag(num_view);
            for (int k = 0; k < seam_flag.size(); k++) {
                seam_flag[k] = false;
            }

            cv::Mat cur_seam_mask =
                cv::Mat::zeros(total_mask.rows, total_mask.cols, CV_8UC1);
            BFS_get_seam_mask(make_int2(i, j), seam_flag, total_mask,
                              total_seam_map, cur_seam_mask, visited);

            if (seam_flag[0]) {
                if (seam_flag.back())
                    seam_masks[0] = cur_seam_mask;
                else if (seam_flag[1])
                    seam_masks[1] = cur_seam_mask;
            } else {
                for (int k = 1; k < seam_flag.size() - 1; k++) {
                    if ((seam_flag[k] == 1) && (seam_flag[k + 1] == 1)) {
                        seam_masks[k + 1] = cur_seam_mask;
                        break;
                    }
                }
            }
            mark_idx++;
        }
    }

    return seam_masks;
}

static inline void allocate_seam_masks_GPU(std::vector<cv::Mat> seam_masks_cpu,
                                           std::vector<uchar*> seam_masks,
                                           int scale) {
    for (int i = 0; i < seam_masks_cpu.size(); i++) {
        cv::Mat mask;
        cv::resize(seam_masks_cpu[i], mask,
                   cv::Size(seam_masks_cpu[i].cols * scale,
                            seam_masks_cpu[i].rows * scale));
        cudaMemcpy(seam_masks[i], mask.ptr<uchar>(0),
                   sizeof(uchar) * seam_masks_cpu[i].rows *
                       seam_masks_cpu[i].cols * scale * scale,
                   cudaMemcpyHostToDevice);
    }
}
#endif