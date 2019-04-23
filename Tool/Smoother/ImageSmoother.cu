#include <driver_types.h>
#include "ImageSmoother.cuh"
#include <stdio.h>
#include <iostream>

__device__ __host__ bool insert2(float *element_entry, int pos, float elem){
    element_entry[pos] = elem;
    return true;
}


// Simple transformation kernel
__global__ void trimmed_mean(float *output, cudaTextureObject_t texObj,
                             int kernel_radius, float trimmed_ratio, int width, int height){
    // Step 1: Calculate normalized texture coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x > width - 1 || y > height - 1){    // Out of range
        return;
    }

    extern __shared__ float sdata[];
    int k = (2 * kernel_radius + 1) * (2 * kernel_radius + 1) * trimmed_ratio + 1;
    //printf("k = %d\n", k);
    const int myId = threadIdx.x + threadIdx.y * blockDim.x;

    float *max_heap_entry = &sdata[2 * myId * k];
    float *min_heap_entry = (float *)&sdata[(2 * myId + 1) * k];

    int2 left_up = make_int2((x - kernel_radius >= 0) ? (x - kernel_radius) : 0,
                             (y - kernel_radius >= 0) ? (y - kernel_radius) : 0);

    int2 right_bottom = make_int2((x + kernel_radius < width) ? (x + kernel_radius) : width - 1,
                                  (y + kernel_radius < height) ? (y + kernel_radius) : height - 1);

    int cnt = (right_bottom.x - left_up.x + 1) * (right_bottom.y - left_up.y + 1);
    //printf("\tcnt = %d\n", cnt);

    k = cnt * trimmed_ratio;
    //printf("k = %d\n", k);

    Max_heap max_heap(k, max_heap_entry);
    Min_heap min_heap(k, min_heap_entry);
    float cur_value;
    float value = 0.0f;

    for(int r = left_up.y; r <= right_bottom.y; ++r){
        for(int c = left_up.x; c <= right_bottom.x; ++c){
            cur_value = tex2D<float>(texObj, float(c) + 0.5, float(r) + 0.5);
            value += cur_value;
            if(!max_heap.full()){
                max_heap.insert(cur_value);
                min_heap.insert(cur_value);
            }else if(cur_value < max_heap.top()){
                max_heap.delete_and_insert(cur_value);
            }else if(cur_value > min_heap.top()){
                min_heap.delete_and_insert(cur_value);
            }
        }

        // Step 2: Read from texture amd write to global memory
        output[y * width + x] = (value - max_heap.sum() - min_heap.sum()) / float(cnt - 2 * k);
    }
}



void ImageSmoother::image_smooth(float *d_array, int kernel_radius, float trimmed_ratio, int width, int height) {
    size_t size = width * height * sizeof(float);

    // Step 1: Allocate CUDA array in device memory
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &floatTex, width, height);

    // Step 2: Copy to device memory
    cudaMemcpyToArray(cuArray, 0, 0, d_array, size, cudaMemcpyDeviceToDevice);

    // Step 3: Define cudaResourceDesc
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Step 4: Define cudaTextureDesc
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;

    // Step 5: Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Step 6: Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // int kernel_radius = 4;
    // float trimmed_ratio = 0.05;
    int heap_size = (2 * kernel_radius + 1) * (2 * kernel_radius + 1) * trimmed_ratio + 1;
    //printf("\nheap_size = %d\n", heap_size);
    trimmed_mean << <dimGrid, dimBlock, sizeof(float) * heap_size * 2 * dimBlock.x * dimBlock.y >> >
                                        (d_array, texObj, kernel_radius, trimmed_ratio, width, height);

    // Finally, clean up
    //  1) Destroy texture object
    cudaDestroyTextureObject(texObj);
    //  2) Free device memory
    cudaFreeArray(cuArray);
}