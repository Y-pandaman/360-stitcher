/*
 * @Author: 姚潘涛
 * @Date: 2024-04-23 19:16:24
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-04-26 17:51:27
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "PanoMain.h"
#include <thread>
int main(int argc, char** argv) {
    // std::thread tp(&panoMain, "../parameters", false);
    std::thread tp(&panoMain, argv[1], (bool)std::atoi(argv[2]));
    tp.join();
}
