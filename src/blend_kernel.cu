#include "blend.h"


__device__ __constant__ float GauKernel[25] = {
    0.0039, 0.0156, 0.0234, 0.0156, 0.0039,
    0.0156, 0.0625, 0.0938, 0.0625, 0.0156,
    0.0234, 0.0938, 0.1406, 0.0938, 0.0234,
    0.0156, 0.0625, 0.0938, 0.0625, 0.0156,
    0.0039, 0.0156, 0.0234, 0.0156, 0.0039};


template<int CHANNEL>
__device__  static inline void weight_product(float *color, short *src, float &aw, float w)
{   
    #pragma unroll
    for (int i=0; i<CHANNEL; i++)
    {
        color[i] = color[i] + (float)src[i] * w; 
    }
    aw += w;
}



__global__
void fast_erode(
    uchar* seam_mask,
    uchar* mask, int radius,
    int height, int width)
{
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel = height * width;
    while(pixelIdx < totalPixel)
    {

        int x = pixelIdx % width;
        int y = pixelIdx / width;

        int remain = 1;

        for (int i=-radius; i<=radius && remain == 1; i++)
        {
            for (int j=-radius; j<=radius; j++)
            {
                int u = x + j, v = y + i;

                if (u < 0 || v < 0 || u >= width || v >= height) continue;

                if (mask[u + v * width] == 0)
                {
                    remain = 0; break;
                }

            }
        }

        seam_mask[pixelIdx] = seam_mask[pixelIdx] * remain;

        pixelIdx += total_thread;
    }
}

__global__ 
void convertBack_kernel(
    short3* src,
    uchar3* dst,
    uchar* mask0, uchar* mask1, uchar* mask2, uchar* mask3, uchar* mask4, uchar* mask5,
    int height, int width)
{
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel = height * width;
    while(pixelIdx < totalPixel)
    {   

        int remian = mask0[pixelIdx]/255 | mask1[pixelIdx]/255 | mask2[pixelIdx]/255 | mask3[pixelIdx]/255 | mask4[pixelIdx]/255 | mask5[pixelIdx]/255;
        dst[pixelIdx].x = remian * clamp((float)src[pixelIdx].x, 0.0f, 32767.0f) / 32767.0f * 255.0f;
        dst[pixelIdx].y = remian * clamp((float)src[pixelIdx].y, 0.0f, 32767.0f) / 32767.0f * 255.0f;
        dst[pixelIdx].z = remian * clamp((float)src[pixelIdx].z, 0.0f, 32767.0f) / 32767.0f * 255.0f;
        pixelIdx += total_thread;
    }   
}

template<int CHANNEL>
__global__
void converTo_kernel(
    uchar* src,
    uchar* mask,
    short* dst,
    float scale,
    int height, int width)
{
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel = height * width;
    while(pixelIdx < totalPixel)
    {   

        #pragma unroll
        for (int i=0; i<CHANNEL; i++)
        {
            dst[CHANNEL*pixelIdx+i] = (float)src[CHANNEL*pixelIdx+i] * scale;
        }

        if (CHANNEL == 3 && mask[pixelIdx] == 0)
        {
            #pragma unroll
            for (int i=0; i<CHANNEL; i++)
            {
                dst[CHANNEL*pixelIdx+i] = -32768;
            }
        }

        pixelIdx += total_thread;
    }
}

/*
ALGORITHM description
The function performs the downsampling step of the Gaussian pyramid construction. 
First, it convolves the source image with the kernel: GauKernel
Then, it downsamples the image by rejecting even rows and columns.

Each thread is responsible for each pixel in high level (low resolution).
*/
template<int CHANNEL>
__global__ void PyrDown_kernel(
    short* src,
    short* dst,
    int height, int width)
{
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel = height * width;
    while(pixelIdx < totalPixel)
    {
        int src_x = 2 * (pixelIdx % width);
        int src_y = 2 * (pixelIdx / width);
        int src_pixelIdx = src_y * (width * 2) + src_x;

        float color[CHANNEL];
        for (int i=0; i<CHANNEL; i++) 
            color[i] = 0;
        float weight = 0.0f;


        #pragma unroll 
        for (int i=0; i<5; i++)
        {
            #pragma unroll 
            for (int j=0; j<5; j++)
            {
                int u = src_x + j - 2;
                int v = src_y + i - 2;

                if (u < 0 || v < 0 || u >= width*2  || v >= height*2) continue;

                // if (src_mask[v*2*width+u] < 128) continue; 

                if (src[CHANNEL*(v*2*width+u)] == -32768) continue;

                weight_product<CHANNEL>(color, src+CHANNEL*(v*2*width+u), weight, GauKernel[i*5+j]);

                // if (u >= 0 && v >= 0 && u < width*2 && v < height*2)
                // {
                //     weight_product<CHANNEL>(color, src+CHANNEL*(v*2*width+u), weight, GauKernel[i*5+j]);
                // }
            
            }
        }
        #pragma unroll 

        if (weight == 0)
        {
        for (int i=0; i<CHANNEL; i++)
            {
                dst[CHANNEL*pixelIdx+i] = -32768;
            }
        }else{
            for (int i=0; i<CHANNEL; i++)
            {
                color[i] = color[i] / weight;
                dst[CHANNEL*pixelIdx+i] = (short)color[i];
            }
        }

        pixelIdx += total_thread;
    }
}

/*
ALGORITHM description
The image is first upsized to twice the original in each dimension, with the 
new (even) rows filled with 0s. Thereafter, a convolution is performed with 
Gaussian filter to approximate the value of the "missing" pixels.

This filter is also normalized to 4, rather rhan 1. (because inserted rows have 0s)

Each thread is responsible for each pixel in low level (high resolution).
*/ 
template<int SIGN>
__global__ void PyrUp_kernel(
    short3* src,
    short3* dst,
    int height, int width) 
{
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel = height * width;
    while(pixelIdx < totalPixel)
    {  

        int x = pixelIdx % width;
        int y = pixelIdx / width;

        float3 color = make_float3(0,0,0);

        float weight = 0;

        #pragma unroll 
        for (int i=0; i<5; i++)
        {
            #pragma unroll 
            for (int j=0; j<5; j++)
            {
                int u = x + j - 2;
                int v = y + i - 2;

                if (u < 0 || v < 0 || u >= width || v >= height) continue;

                if (src[v/2*width/2+u/2].x == -32768) continue;

                if (u % 2 ==0 && v % 2 == 0)
                {
                    weight_product<3>(&color.x, (short*)(src+(v/2*width/2+u/2)), weight, 4*GauKernel[i*5+j]);
                }
            }
        }

        if (weight != 0)
        {
            color = color / weight;

            color.x = (float)dst[pixelIdx].x + SIGN * color.x;
            color.y = (float)dst[pixelIdx].y + SIGN * color.y;
            color.z = (float)dst[pixelIdx].z + SIGN * color.z;

            color = clamp(color, -32768.0f, 32767.0f);
        }

        dst[pixelIdx].x = color.x;
        dst[pixelIdx].y = color.y;
        dst[pixelIdx].z = color.z;

        pixelIdx += total_thread;
    }
}



// [NOTE] ******* 输出到I1 ***************
__global__
void WeightedBlend_kernel(
    short3* I0, short* mask0, 
    short3* I1, short* mask1,
    short3* I2, short* mask2,
    short3* I3, short* mask3,
    short3* I4, short* mask4,
    short3* I5, short* mask5,
    int height, int width)
{
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel = height * width;
    while(pixelIdx < totalPixel)
    {   
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        float w0 = (float)mask0[pixelIdx] / 32767.0f;
        float w1 = (float)mask1[pixelIdx] / 32767.0f;
        float w2 = (float)mask2[pixelIdx] / 32767.0f;
        float w3 = (float)mask3[pixelIdx] / 32767.0f;
        float w4 = (float)mask4[pixelIdx] / 32767.0f;
        float w5 = (float)mask5[pixelIdx] / 32767.0f;
        float total_w = w0 + w1 + w2 + w3 + w4 + w5;

        if (total_w > 0)
        {   
            I1[pixelIdx].x = (short)(((float)I0[pixelIdx].x * w0 + (float)I1[pixelIdx].x * w1 + (float)I2[pixelIdx].x * w2 + (float)I3[pixelIdx].x * w3 + (float)I4[pixelIdx].x * w4 + (float)I5[pixelIdx].x * w5) / total_w);
            I1[pixelIdx].y = (short)(((float)I0[pixelIdx].y * w0 + (float)I1[pixelIdx].y * w1 + (float)I2[pixelIdx].y * w2 + (float)I3[pixelIdx].y * w3 + (float)I4[pixelIdx].y * w4 + (float)I5[pixelIdx].y * w5) / total_w);
            I1[pixelIdx].z = (short)(((float)I0[pixelIdx].z * w0 + (float)I1[pixelIdx].z * w1 + (float)I2[pixelIdx].z * w2 + (float)I3[pixelIdx].z * w3 + (float)I4[pixelIdx].z * w4 + (float)I5[pixelIdx].z * w5) / total_w);
        }else{
            I1[pixelIdx].x = 0;
            I1[pixelIdx].y = 0;
            I1[pixelIdx].z = 0;
        }

        pixelIdx += total_thread;
    }
}

std::vector<SeamImageGPU> LaplacianPyramid(
    CylinderImageGPU img, 
    uchar* seam_mask,
    int levels)
{
    std::vector<SeamImageGPU> pyr_imgs;

    int num_thread = 512;
    int height = img.height, width = img.width;
    int num_block = min(65535, (height*width + num_thread - 1) / num_thread);

    short3 *current_img;
    cudaMalloc((void**)&current_img, sizeof(short3)*height*width);
    short *current_mask;
    cudaMalloc((void**)&current_mask, sizeof(short)*height*width);

    converTo_kernel<3><<<num_block, num_thread>>>(
        (uchar*)img.image, img.mask, (short*)current_img, 32767.0f / 255.0f, height, width);
    converTo_kernel<1><<<num_block, num_thread>>>(
        seam_mask, NULL, current_mask, 32767.0f / 255.0f, height, width);

    pyr_imgs.emplace_back(SeamImageGPU(current_img, current_mask, height, width));

    for (int i=1; i<levels; i++)
    {
        height /= 2; width /= 2;
        num_block = min(65535, (height*width + num_thread - 1) / num_thread);
        
        short3 *_down_scale_img;
        cudaMalloc((void**)&_down_scale_img, sizeof(short3)*height*width);
        short *_down_scale_mask;
        cudaMalloc((void**)&_down_scale_mask, sizeof(short)*height*width);

        PyrDown_kernel<3><<<num_block, num_thread>>>(
            (short*)current_img, (short*)_down_scale_img, height, width);
        PyrDown_kernel<1><<<num_block, num_thread>>>(
            current_mask, _down_scale_mask, height, width);

        pyr_imgs.emplace_back(SeamImageGPU(_down_scale_img, _down_scale_mask, height, width));
        
        current_img = _down_scale_img;
        current_mask = _down_scale_mask;
    }

    for (int i=0; i<levels-1; i++)
    {
        int num_block = min(65535, (pyr_imgs[i].height*pyr_imgs[i].width + num_thread - 1) / num_thread);
        PyrUp_kernel<-1><<<num_block, num_thread>>>(
            pyr_imgs[i+1].image,
            pyr_imgs[i].image,
            pyr_imgs[i].height, pyr_imgs[i].width);
    }


    // for (int i=0; i<pyr_imgs.size(); i++)
    // {
    //     short3 *gau_img;
    //     short* gau_img_mask;
    //     int h = pyr_imgs[i].height, w = pyr_imgs[i].width;
    //     cudaHostAlloc((void**)&gau_img,sizeof(short3)*h*w,cudaHostAllocDefault);
    //     cudaHostAlloc((void**)&gau_img_mask,sizeof(short)*h*w,cudaHostAllocDefault);
    //     cudaMemcpy(gau_img, pyr_imgs[i].image, sizeof(short3)*h*w, cudaMemcpyDeviceToHost);
    //     cudaMemcpy(gau_img_mask, pyr_imgs[i].mask, sizeof(short)*h*w, cudaMemcpyDeviceToHost);

    
    //     cv::Mat out(h, w, CV_16SC3, gau_img);
    //     cv::Mat out_mask(h, w, CV_16SC1, gau_img_mask);

    //     out.convertTo(out, CV_8UC3, 255.0f / 32767.0f);
    //     out_mask.convertTo(out_mask, CV_8UC1, 255.0f / 32767.0f);

    //     cv::imwrite("out_0_level_"+std::to_string(i)+"_up.png", out);
    //     cv::imwrite("out_0_level_mask_"+std::to_string(i)+".png", out_mask);
    // }

    return pyr_imgs;
}


std::vector<SeamImageGPU> BlendPyramid(
    std::vector<SeamImageGPU> pyr_img0,
    std::vector<SeamImageGPU> pyr_img1,
    std::vector<SeamImageGPU> pyr_img2,
    std::vector<SeamImageGPU> pyr_img3,
    std::vector<SeamImageGPU> pyr_img4,
    std::vector<SeamImageGPU> pyr_img5)
{
    int num_thread = 512;

    std::vector<SeamImageGPU> blendedPyramid;

    for (int i=0; i<pyr_img0.size(); i++)
    {
        int num_block = min(65535, (pyr_img0[i].height*pyr_img0[i].width + num_thread - 1) / num_thread);
        WeightedBlend_kernel<<<num_block, num_thread>>>(
            pyr_img0[i].image, pyr_img0[i].mask, 
            pyr_img1[i].image, pyr_img1[i].mask, 
            pyr_img2[i].image, pyr_img2[i].mask, 
            pyr_img3[i].image, pyr_img3[i].mask,
            pyr_img4[i].image, pyr_img4[i].mask,
            pyr_img5[i].image, pyr_img5[i].mask, 
            pyr_img0[i].height, pyr_img0[i].width);
        
        blendedPyramid.emplace_back(SeamImageGPU(pyr_img1[i].image, NULL, pyr_img1[i].height, pyr_img1[i].width));
    }

    // for (int i=0; i<blendedPyramid.size(); i++)
    // {
    //     short3 *gau_img;
    //     int h = blendedPyramid[i].height, w = blendedPyramid[i].width;
    //     cudaHostAlloc((void**)&gau_img,sizeof(short3)*h*w,cudaHostAllocDefault);
    //     cudaMemcpy(gau_img, blendedPyramid[i].image, sizeof(short3)*h*w, cudaMemcpyDeviceToHost);
    //     cv::Mat out(h, w, CV_16SC3, gau_img);
    //     out.convertTo(out, CV_8UC3, 255.0f / 32767.0f);
    //     cv::imwrite("blend_0_level_"+std::to_string(i)+"_lap.png", out);
    // }

    return blendedPyramid;
}

void CollapsePyramid(std::vector<SeamImageGPU> blendedPyramid,
                     std::vector<CylinderImageGPU> cylImages)
{
    
    int num_thread = 512, num_block = 0;

    for(int i=blendedPyramid.size()-2; i>=0; i--)
    {
        num_block = min(65535, (blendedPyramid[i].height*blendedPyramid[i].width + num_thread - 1) / num_thread);
        PyrUp_kernel<1><<<num_block, num_thread>>>(
            blendedPyramid[i+1].image,
            blendedPyramid[i].image,
            blendedPyramid[i].height, blendedPyramid[i].width);
    }

    convertBack_kernel<<<num_block, num_thread>>>(
        blendedPyramid[0].image, cylImages[1].image, 
        cylImages[0].mask, cylImages[1].mask, cylImages[2].mask, cylImages[3].mask, cylImages[4].mask, cylImages[5].mask,
        blendedPyramid[0].height, blendedPyramid[0].width);
}



__host__ 
void MultiBandBlend_cuda(
    std::vector<CylinderImageGPU> cylImages,
    std::vector<uchar*> seam_masks)
{   

    int levels = 5;

    std::vector<SeamImageGPU> pyr_img0 = LaplacianPyramid(cylImages[0], seam_masks[0], levels);
    std::vector<SeamImageGPU> pyr_img1 = LaplacianPyramid(cylImages[1], seam_masks[1], levels);
    std::vector<SeamImageGPU> pyr_img2 = LaplacianPyramid(cylImages[2], seam_masks[2], levels);
    std::vector<SeamImageGPU> pyr_img3 = LaplacianPyramid(cylImages[3], seam_masks[3], levels);
    std::vector<SeamImageGPU> pyr_img4 = LaplacianPyramid(cylImages[4], seam_masks[4], levels);
    std::vector<SeamImageGPU> pyr_img5 = LaplacianPyramid(cylImages[5], seam_masks[5], levels);


    std::vector<SeamImageGPU> blendedPyramid = BlendPyramid(pyr_img0, pyr_img1, pyr_img2, pyr_img3, pyr_img4, pyr_img5);

    CollapsePyramid(blendedPyramid, cylImages);

    for (int i=0; i<pyr_img0.size(); i++)
    {
        pyr_img0[i].clear();
        pyr_img1[i].clear();
        pyr_img2[i].clear();
        pyr_img3[i].clear();
        pyr_img4[i].clear();
        pyr_img5[i].clear();
    }
    
}