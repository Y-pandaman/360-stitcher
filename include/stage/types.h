#ifndef UNION_DATA_TYPE_HPP
#define UNION_DATA_TYPE_HPP

#include "config.h"
#include "stdint.h"

union Uint16_bytes {
    uint16_t data;
    uint8_t bytes[2];
};
union Uint32_bytes {
    uint32_t data;
    uint8_t bytes[4];
};
union Uint64_bytes {
    uint64_t data;
    uint8_t bytes[8];
};

union Int16_bytes {
    int16_t data;
    uint8_t bytes[2];
};
union Int32_bytes {
    int32_t data;
    uint8_t bytes[4];
};
union Int64_bytes {
    int64_t data;
    uint8_t bytes[8];
};

union Float_bytes {
    float data;
    uint8_t bytes[sizeof(float)];
};
union Double_bytes {
    double data;
    uint8_t bytes[sizeof(double)];
};

struct YoloKernel {
    int width;
    int height;
    float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection {
    float bbox[4];  // center_x center_y w h
    float conf;     // bbox_conf * cls_conf
    float class_id;
    float mask[32];
};
#endif //UNION_DATA_TYPE_HPP
