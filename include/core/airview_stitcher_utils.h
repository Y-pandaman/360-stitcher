#ifndef __AIRVIEW_STITCHER__UTILS__
#define __AIRVIEW_STITCHER__UTILS__

#include "common/cylinder_stitcher.cuh"

#define MASK_MAX 255

typedef std::pair<int, int2> myPair;

__device__ __constant__ static float GauKernel[25] = {
    0.0039, 0.0156, 0.0234, 0.0156, 0.0039, 0.0156, 0.0625, 0.0938, 0.0625,
    0.0156, 0.0234, 0.0938, 0.1406, 0.0938, 0.0234, 0.0156, 0.0625, 0.0938,
    0.0625, 0.0156, 0.0039, 0.0156, 0.0234, 0.0156, 0.0039};

const int2 search_seq[8] = {
    make_int2(1, 0),  make_int2(-1, 0), make_int2(0, -1), make_int2(0, 1),
    make_int2(-1, 1), make_int2(1, 1),  make_int2(1, -1), make_int2(-1, -1)};

/**
 * 自定义比较器结构体cmp
 * 该结构体重载了()运算符，作为一个比较函数，用于比较两个对象的“first”成员。
 *
 * @tparam T 比较对象的类型1
 * @tparam U 比较对象的类型2
 * @param left 要比较的第一个对象
 * @param right 要比较的第二个对象
 * @return
 * 如果left的"first"成员大于right的"first"成员，则返回true；否则返回false。
 */
struct cmp {
    template <typename T, typename U>
    bool operator()(T const& left, U const& right) {
        // 比较两个对象的"first"成员，并返回比较结果
        if (left.first > right.first)
            return true;
        return false;
    }
};

/**
 * @brief 对三维点进行基于homograph的变形。
 *
 * @param p 待变形的三维点，其中x、y、z分别表示点的坐标。
 * @param H 变形所使用的homograph矩阵，是一个3x3的矩阵。
 * @return float3 返回变形后的三维点坐标。
 */
static inline float3 homographBased_warp(float3 p, cv::Mat H) {
    float3 warped_p;   // 用于存储变形后的点

    // 通过homograph矩阵对点进行变形
    warped_p.x = H.at<float>(0, 0) * p.x + H.at<float>(0, 1) * p.y +
                 H.at<float>(0, 2) * p.z;
    warped_p.y = H.at<float>(1, 0) * p.x + H.at<float>(1, 1) * p.y +
                 H.at<float>(1, 2) * p.z;
    warped_p.z = H.at<float>(2, 0) * p.x + H.at<float>(2, 1) * p.y +
                 H.at<float>(2, 2) * p.z;

    return warped_p;
}

/**
 * 检查给定像素点是否位于图像边界或其周围像素值低于特定阈值。
 *
 * @param src 输入的cv::Mat图像，要求为8位单通道图像。
 * @param i 像素点的行索引。
 * @param j 像素点的列索引。
 * @return
 * 如果指定像素点位于图像边界或其周围8邻域内存在像素值小于128的点，则返回true；否则返回false。
 */
static inline bool is_boundary(cv::Mat src, int i, int j) {
    // 检查像素点是否位于图像的边界
    if (i == 0 || j == 0 || i == src.rows - 1 || j == src.cols - 1)
        return true;

    // 遍历像素点的8邻域，检查是否存在像素值小于128的点
    for (int k = 0; k < 8; k++) {
        if (src.at<uchar>(i + search_seq[k].x, j + search_seq[k].y) < 128)
            return true;
    }
    return false;
}

/**
 * 深度优先搜索收集重叠点
 *
 * 该函数通过深度优先搜索（DFS）的方式，从给定的位置开始，收集所有与给定标记号码一致的重叠点。
 *
 * @param loc         当前搜索的位置，是一个包含x和y坐标的整数对。
 * @param mark_number 需要搜索的标记号码。
 * @param marks       包含标记信息的cv::Mat矩阵，用于确定位置上的标记号码。
 * @param overlap_points 收集到的重叠点集合，是一个整数对的向量。
 * @param visited     记录已访问位置的cv::Mat矩阵，用于避免重复访问。
 */
static inline void DFS_collect_overlap_point(int2 loc, int mark_number,
                                             cv::Mat marks,
                                             std::vector<int2>& overlap_points,
                                             cv::Mat& visited) {
    overlap_points.emplace_back(loc);   // 将当前位置加入到重叠点集合中
    visited.at<uchar>(loc.x, loc.y) = 1;   // 标记当前位置为已访问

    // 遍历当前位置的8个相邻位置
    for (int k = 0; k < 8; k++) {
        int2 next_loc = loc + search_seq[k];   // 计算下一个位置

        // 跳过越界的点
        if (next_loc.x < 0 || next_loc.y < 0 || next_loc.x >= visited.rows ||
            next_loc.y >= visited.cols)
            continue;

        // 跳过已访问的点
        if (visited.at<uchar>(next_loc.x, next_loc.y) == 1)
            continue;

        // 跳过标记号码不匹配的点
        if (marks.at<uchar>(next_loc.x, next_loc.y) != mark_number)
            continue;

        // 递归搜索符合条件的相邻点
        DFS_collect_overlap_point(next_loc, mark_number, marks, overlap_points,
                                  visited);
    }
}

/**
 * 决定重叠区域的起始点和结束点
 *
 * 本函数通过分析重叠掩膜和两个掩膜图像，标识出重叠区域的边缘，并选择其中距离图像中心较近的一点作为起始点，远的一点作为结束点。
 *
 * @param overlap_mask 重叠掩膜图像，用于标识两个图像重叠的区域。
 * @param mask0 第一个掩膜图像。
 * @param mask1 第二个掩膜图像。
 * @return 返回一个包含起始点和结束点坐标的int4类型（x1, y1, x2, y2），其中(x1,
 * y1)是起始点，(x2, y2)是结束点。
 */
static inline int4 decide_start_end(cv::Mat overlap_mask, cv::Mat mask0,
                                    cv::Mat mask1) {
    // STEP1: 标记重叠边缘区域
    cv::Mat marks =
        cv::Mat::zeros(overlap_mask.rows, overlap_mask.cols, CV_8UC1);
    for (int i = 0; i < overlap_mask.rows; i++) {
        for (int j = 0; j < overlap_mask.cols; j++) {
            if (overlap_mask.at<uchar>(i, j) != 255)
                continue;
            // 检查图像边界
            if (is_boundary(overlap_mask, i, j) && is_boundary(mask0, i, j) &&
                is_boundary(mask1, i, j)) {
                marks.at<uchar>(i, j) = 255;
            }
        }
    }

    // STEP2: 收集重叠边缘点
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
            // 收集重叠区域的起始和结束点
            DFS_collect_overlap_point(make_int2(i, j), 255, marks, temp,
                                      visited);
            overlap_points.emplace_back(temp);
        }
    }

    // 确保找到了至少两个重叠边缘点
    assert(overlap_points.size() >= 2);

    // 选择中点作为起始点和结束点
    int2 start = overlap_points[0][overlap_points[0].size() / 2];
    int2 end   = overlap_points[1][overlap_points[1].size() / 2];

    // 计算起始点和结束点到图像中心的距离，选择距离较近的作为起始点
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

/**
 * 使用Dijkstra算法进行缝合搜索
 *
 * @param start 起始点的二维坐标
 * @param end 结束点的二维坐标
 * @param diff 表示差异的图像，用于计算权重
 * @param overlap_mask 重叠区域的掩码图像，标识哪些像素可以作为路径的一部分
 * @param seam_map 存储缝合路径的映射图像
 * @param seam_line 表示缝合线的图像
 * @param seam_mark 用于标记缝合像素的值
 */
static inline void Dijkstra_search_seam(int2 start, int2 end, cv::Mat diff,
                                        cv::Mat overlap_mask, cv::Mat& seam_map,
                                        cv::Mat& seam_line, int seam_mark) {
    // 初始化visited矩阵，用于标记顶点是否被访问
    cv::Mat visited = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
    // 初始化距离矩阵D，记录起点到各顶点的最短距离
    cv::Mat D = cv::Mat::ones(diff.rows, diff.cols, CV_32FC1) * 100000000;
    // 初始化路径矩阵path，记录各顶点的前置顶点
    cv::Mat path = cv::Mat::ones(diff.rows, diff.cols, CV_32FC2) * -1;

    // 设置起始点的距离为0
    D.at<float>(start.x, start.y) = 0;

    // 使用优先队列来存储顶点及其距离
    std::priority_queue<myPair, std::vector<myPair>, cmp> Q;

    Q.emplace(0, start);

    // Dijkstra算法的主要循环
    while (!Q.empty()) {
        int2 loc = Q.top().second;
        Q.pop();
        // 跳过已访问的顶点
        while (!Q.empty() && visited.at<uchar>(loc.x, loc.y) == 1) {
            loc = Q.top().second;
            Q.pop();
        }

        if (visited.at<uchar>(loc.x, loc.y) == 1) {
            break;   // 所有顶点均已访问
        }

        // 标记当前顶点为已访问
        visited.at<uchar>(loc.x, loc.y) = 1;

        // 遍历当前顶点的邻接顶点
        for (auto k = 0; k < 4; k++) {
            int2 next_loc = loc + search_seq[k];

            // 跳过不在图像范围内、已被访问或不在重叠区域的像素
            if (next_loc.x < 0 || next_loc.y < 0 || next_loc.x >= diff.rows ||
                next_loc.y >= diff.cols)
                continue;
            if (seam_map.at<uchar>(next_loc.x, next_loc.y) != 255)
                continue;
            if (overlap_mask.at<uchar>(next_loc.x, next_loc.y) == 0)
                continue;
            if (visited.at<uchar>(next_loc.x, next_loc.y) == 1)
                continue;
            // 计算权重
            float w =
                ((float)diff.at<ushort>(next_loc.x, next_loc.y) / 65535.0f);

            // 更新距离和路径
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

    // 确定最短路径的终点
    float min_dis = 100000000.0f;
    int2 end_pts  = end;
    // 在重叠区域的边界上寻找距离起点最近的点作为终点
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

    // 根据路径矩阵回溯缝合路径，并标记在seam_map和seam_line中
    while (true) {
        seam_map.at<uchar>(end_pts.x, end_pts.y)  = seam_mark;
        seam_line.at<uchar>(end_pts.x, end_pts.y) = 255;

        if (end_pts.x == start.x && end_pts.y == start.y)
            break;

        float2 prev_loc = path.at<float2>(end_pts.x, end_pts.y);
        end_pts.x       = (int)prev_loc.x;
        end_pts.y       = (int)prev_loc.y;
    }

    // cv::imshow("seam_map", seam_map);
    // cv::waitKey(1);
}

/**
 * 执行缝合搜索，使用Dijkstra算法找到从起点到终点的最小代价路径。
 *
 * @param endPts 包含起点和终点的int4对象，(x, y)为起点坐标，(z, w)为终点坐标。
 * @param overlap_mask 重叠区域的掩码图像，用于限制搜索区域。
 * @param diff 输入图像的差异图像，用于计算像素间的差异代价。
 * @param seam_map 存储搜索路径上每个像素的最小代价。
 * @param seam_line 存储搜索到的缝合线的像素坐标。
 * @param seam_mark 用于标记缝合线的特殊值。
 */
static inline void seam_search(int4 endPts, cv::Mat overlap_mask, cv::Mat diff,
                               cv::Mat& seam_map, cv::Mat& seam_line,
                               int seam_mark) {
    // 初始化起点和终点
    int2 start = make_int2(endPts.x, endPts.y);
    int2 end   = make_int2(endPts.z, endPts.w);
    // 执行Dijkstra算法进行缝合搜索
    Dijkstra_search_seam(start, end, diff, overlap_mask, seam_map, seam_line,
                         seam_mark);
}

/**
 * 使用BFS算法获取 seams 的掩码。
 *
 * @param start 起始点的坐标（x, y）。
 * @param seam_flag 用于标记当前区域相关的两条seam。
 * @param total_mask 总体掩码，确定搜索的边界。
 * @param seam_map seam映射图，已存在的seam位置用特定值标记。
 * @param seam_mask 输出参数，生成的seam掩码。
 * @param visited 访问标记，用于避免重复访问。
 */
static inline void BFS_get_seam_mask(int2 start, std::vector<bool>& seam_flag,
                                     cv::Mat total_mask, cv::Mat seam_map,
                                     cv::Mat& seam_mask, cv::Mat& visited) {
    std::queue<int2> Q;

    Q.push(start);
    visited.at<uchar>(start.x, start.y) = 1;
    // 标记起始点为已访问，并在seam_mask上做标记
    seam_mask.at<uchar>(start.x, start.y) = 255;

    // 使用BFS遍历图像来标记seam
    while (!Q.empty()) {
        int2 loc = Q.front();
        Q.pop();

        // 跳过已标记为seam的点
        if (seam_map.at<uchar>(loc.x, loc.y) != 255) {
            continue;
        }

        // 遍历当前点的四个相邻点
        for (int k = 0; k < 4; k++) {
            int2 next_loc = loc + search_seq[k];

            // 跳过越界的点
            if (next_loc.x < 0 || next_loc.y < 0 ||
                next_loc.x >= visited.rows || next_loc.y >= visited.cols)
                continue;

            // 跳过不在total_mask内的点
            if (total_mask.at<uchar>(next_loc.x, next_loc.y) < 128)
                continue;

            // 如果相邻点已经在seam_map上被标记，标记对应的seam_flag
            if (seam_map.at<uchar>(next_loc.x, next_loc.y) != 255) {
                seam_flag[seam_map.at<uchar>(next_loc.x, next_loc.y)] = true;
            }

            // 跳过已访问的点
            if (visited.at<uchar>(next_loc.x, next_loc.y) == 1)
                continue;

            Q.push(next_loc);   // 将未访问的相邻点加入队列

            visited.at<uchar>(next_loc.x, next_loc.y) = 1;   // 标记为已访问
            // 在seam_mask上做标记
            seam_mask.at<uchar>(next_loc.x, next_loc.y) = 255;
        }
    }
}

/**
 * 生成每个视图的接缝掩膜。
 *
 * @param total_mask 总的掩膜图像，用于确定哪些像素属于特定视图的区域。
 * @param total_seam_map 总的接缝图，标识了像素属于哪个接缝。
 * @param num_view 视图的数量。
 * @return 返回一个包含每个视图接缝掩膜的向量。
 */
static inline std::vector<cv::Mat>
gen_seam_mask(cv::Mat total_mask, cv::Mat total_seam_map, int num_view) {
    std::vector<cv::Mat> seam_masks(num_view);   // 为每个视图分配接缝掩膜的空间

    cv::Mat visited =
        cv::Mat::zeros(total_mask.rows, total_mask.cols,
                       CV_8UC1);   // 创建一个标记矩阵，用于标记已经访问过的像素

    int mark_idx = 0;
    for (int i = 0; i < total_mask.rows; i++) {
        for (int j = 0; j < total_mask.cols; j++) {
            // 跳过接缝图中已定义的像素、已访问过的像素和掩膜值小于128的像素
            if (total_seam_map.at<uchar>(i, j) != 255)
                continue;
            if (visited.at<uchar>(i, j) == 1)
                continue;
            if (total_mask.at<uchar>(i, j) < 128)
                continue;

            // 标记哪个视图的接缝通过当前像素
            std::vector<bool> seam_flag(num_view);
            for (uint64_t k = 0; k < seam_flag.size(); k++) {
                seam_flag[k] = false;
            }

            // 创建当前接缝掩膜
            cv::Mat cur_seam_mask =
                cv::Mat::zeros(total_mask.rows, total_mask.cols, CV_8UC1);
            // 使用BFS算法获取接缝掩膜
            BFS_get_seam_mask(make_int2(i, j), seam_flag, total_mask,
                              total_seam_map, cur_seam_mask, visited);

            // 根据seam_flag确定哪个视图的接缝掩膜被更新
            if (seam_flag[0]) {
                if (seam_flag.back())
                    seam_masks[0] = cur_seam_mask;
                else if (seam_flag[1])
                    seam_masks[1] = cur_seam_mask;
            } else {
                for (uint64_t k = 1; k < seam_flag.size() - 1; k++) {
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

/**
 * @brief 分配GPU上的接缝掩码内存
 *
 * 该函数将CPU上存储的接缝掩码图像放大到指定的尺度，并将它们传输到GPU内存中。
 *
 * @param seam_masks_cpu
 * CPU上存储的原始接缝掩码图像的向量。这些图像将被放大并传输到GPU。
 * @param seam_masks
 * GPU上接缝掩码图像的指针向量，函数执行后将包含放大并传输到GPU上的掩码指针。
 * @param scale 放大接缝掩码图像的尺度因子。
 */
static inline void allocate_seam_masks_GPU(std::vector<cv::Mat> seam_masks_cpu,
                                           std::vector<uchar*> seam_masks,
                                           int scale) {
    // 遍历每一张接缝掩码图像，将其放大并传输到GPU
    for (uint64_t i = 0; i < seam_masks_cpu.size(); i++) {
        cv::Mat mask;   // 用于存储放大的掩码图像
        // 将原始掩码图像放大到指定尺度
        cv::resize(seam_masks_cpu[i], mask,
                   cv::Size(seam_masks_cpu[i].cols * scale,
                            seam_masks_cpu[i].rows * scale));
        // 将放大的掩码图像数据从CPU内存复制到GPU内存
        cudaMemcpy(seam_masks[i], mask.ptr<uchar>(0),
                   sizeof(uchar) * seam_masks_cpu[i].rows *
                       seam_masks_cpu[i].cols * scale * scale,
                   cudaMemcpyHostToDevice);
    }
}
#endif
