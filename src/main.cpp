#include "PanoMain.h"
#include <thread>
int main() {
    std::thread tp(&panoMain, "./parameters", false);
    tp.join();
}

