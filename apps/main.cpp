/*
 * @Author: 姚潘涛
 * @Date: 2024-04-23 19:16:24
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-04-28 09:50:13
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "PanoMain.h"
#include <thread>
int main(int argc, char** argv) {
// int main() {
    // std::thread tp(&panoMain, "../parameters", false);
    std::thread tp(&panoMain, argv[1], (bool)std::atoi(argv[2]));
    tp.join();
}
