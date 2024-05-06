/*
 * @Author: 姚潘涛
 * @Date: 2024-04-23 19:16:24
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-06 20:15:51
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "core/pano_main.h"
#include <thread>
int main(int argc, char** argv) {
    std::thread tp(&panoMain, argv[1], (bool)std::atoi(argv[2]));
    tp.join();
}
