//
// Created by feng on 19-4-22.
//

#ifndef SOLARENERGY_CHIER_IMAGESMOOTHER_CUH
#define SOLARENERGY_CHIER_IMAGESMOOTHER_CUH

#include <stdio.h>
#include <stdio.h>
#include <string>
#include "heap.cuh"

__device__ __host__ bool insert2(float *element_entry, int pos, float elem);

__global__ void trimmed_mean(float *d_output, cudaTextureObject_t texObj,
                            int kernel_radius, float trimmed_ratio, int width, int height);

__global__ void trimmed_gaussian(float *d_output, cudaTextureObject_t texObj,
                            int kernel_radius, float trimmed_ratio, int width, int height, float sigma);

class ImageSmoother{
public:
    static void image_smooth(float *d_array, int kernel_radius, float trimmed_ratio, int width, int height);
    static void image_smooth(float *d_array, int kernel_radius, float trimmed_ratio, int width, int height, float sigma);
    // input as well as output
};



#endif //SOLARENERGY_CHIER_IMAGESMOOTHER_CUH