#include <driver_types.h>
#include "ImageSmoother.cuh"
#include <stdio.h>
#include <iostream>
#include "global_constant.h"
#include "math.h"
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
    // shared memory
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
    }
    // Step 2: Read from texture amd write to global memory
    output[y * width + x] = (value - max_heap.sum() - min_heap.sum()) / float(cnt - 2 * k);
}


//__global__ void trimmed_gaussian(float *output, cudaTextureObject_t texObj,
//                                             int kernel_radius, float trimmed_ratio, int width, int height, float sigma){
//    // Step 1: Calculate normalized texture coordinates
//    const int x = blockIdx.x * blockDim.x + threadIdx.x;
//    const int y = blockIdx.y * blockDim.y + threadIdx.y;
//    if(x > width - 1 || y > height - 1){    // Out of range
//        return;
//    }
//    //printf("\n------------------------------------------------------------------not wrong here\n");
//    // shared memory (Dynamic allocation memory)
//    extern __shared__ float sdata[];
//    extern __shared__ float sk[];
//    int k = (2 * kernel_radius + 1) * (2 * kernel_radius + 1) * trimmed_ratio + 1;
//
//    const int myId = threadIdx.x + threadIdx.y * blockDim.x;
//
//    float *max_heap_entry = &sdata[2 * myId * k];
//    float *min_heap_entry = (float *)&sdata[(2 * myId + 1) * k];
//
//    float *max_k_heap_entry = (float *)&sk[(2 * myId) * k];
//    float *min_k_heap_entry = (float *)&sk[(2 * myId + 1) * k];
//
//    int2 left_up = make_int2((x - kernel_radius >= 0) ? (x - kernel_radius) : 0,
//                             (y - kernel_radius >= 0) ? (y - kernel_radius) : 0);
//
//    int2 right_bottom = make_int2((x + kernel_radius < width) ? (x + kernel_radius) : width - 1,
//                                  (y + kernel_radius < height) ? (y + kernel_radius) : height - 1);
//
//    int cnt = (right_bottom.x - left_up.x + 1) * (right_bottom.y - left_up.y + 1);
//
//    k = cnt * trimmed_ratio;
//
//    // Store the ratio min values and max values
//    Max_heap max_heap(k, max_heap_entry);
//    Min_heap min_heap(k, min_heap_entry);
//    // Store their k_value
//    Min_heap max_k_heap(k, max_k_heap_entry);
//    Max_heap min_k_heap(k, min_k_heap_entry);
//
//    float2 cur_value;
//    float k_sum = 0.0f;
//    float value_sum = 0.0f;
//    int center_x = (right_bottom.x - left_up.x)/2;
//    int center_y = (right_bottom.y - left_up.y)/2;
//    //printf("\ncenter_x, center_y: %d, %d", center_x, center_y);
//
//    float min_heap_sum = 0.0f;
//    float max_heap_sum = 0.0f;
//    float min_k_sum = 0.0f;
//    float max_k_sum = 0.0f;
//
//    for(int r = left_up.y; r <= right_bottom.y; ++r){
//        for(int c = left_up.x; c <= right_bottom.x; ++c){
//            cur_value.x = tex2D<float>(texObj, float(c) + 0.5, float(r) + 0.5);
//
//            cur_value.y = (1/(2 * MATH_PI * sigma * sigma)) * exp(-((r - y) * (r - y) + (c - x) * (c - x)) / (2 * sigma * sigma));
//            //cur_value.y = (1/(2 * MATH_PI * sigma * sigma)) * exp(-((r - center_y) * (r - center_y) + (c - center_x) * (c - center_x)) / (2 * sigma * sigma));
//            //printf("\nr - center_y: %d",r - center_y);
//            //printf("\ncur_value.x:%f",cur_value.x);
//            //printf("\ncur_value.y: %f",cur_value.y);
//            value_sum += cur_value.x * cur_value.y;
//            k_sum += cur_value.y;
//
//            if(!max_heap.full()){
//                max_heap.insert(cur_value.x);
//                min_heap.insert(cur_value.x);
//                max_k_heap.insert(cur_value.y);
//                min_k_heap.insert(cur_value.y);
//            }else if(cur_value.x < max_heap.top()){
//                max_heap.delete_and_insert(cur_value.x);
//                max_k_heap.delete_and_insert(cur_value.y);
//            }else if(cur_value.x > min_heap.top()){
//                min_heap.delete_and_insert(cur_value.x);
//                min_k_heap.delete_and_insert(cur_value.y);
//            }
//        }
//    }
//
//    min_k_sum = min_k_heap.sum();
//
//
//
//    printf("\n----min_k_sum: %f", min_k_sum);
//    printf("\n----min_k_heap.size(): %d", min_k_heap.size());
//    max_k_sum = max_k_heap.sum();
//    printf("\n(max_k_sum): %f",(max_k_sum));
//    printf("\nmax_k_heap.size(): %d", max_k_heap.size());
//
//    while(!max_heap.empty()){
//        max_heap_sum += max_k_heap.top() * max_heap.top();
//        max_k_heap.delete_t();
//        max_heap.delete_t();
//        //printf("\nmax_heap_size: %d", max_heap.size());
//    }
//
//
//
//    while(!min_heap.empty()){
//        min_heap_sum += min_k_heap.top() * min_heap.top();
//        min_k_heap.delete_t();
//        min_heap.delete_t();
//        //printf("\nmin_heap_size: %d", min_heap.size());
//    }
//
//    // Step 2: Read from texture amd write to global memory
//    //output[y * width + x] = (value_sum - max_heap_sum - min_heap_sum) ;
//    //printf("\n(k_sum): %f",(k_sum));
//
//    //printf("\n(k_sum - min_k_sum - max_k_sum): %f",(k_sum - min_k_sum - max_k_sum));
//    output[y * width + x] = (value_sum - max_heap_sum - min_heap_sum) / (k_sum - min_k_sum - max_k_sum);
//    //printf("\noutput: %f",output[y * width + x]);
//    //printf("\nk_sum - max_k_heap.sum() - min_k_heap.sum(): %f",k_sum - max_k_heap.sum() - min_k_heap.sum());
//}



__global__ void trimmed_gaussian(float *output, cudaTextureObject_t texObj,
                                             int kernel_radius, float trimmed_ratio, int width, int height, float sigma){
    // Step 1: Calculate normalized texture coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x > width - 1 || y > height - 1){    // Out of range
        return;
    }
    //printf("\n------------------------------------------------------------------not wrong here\n");
    // shared memory (Dynamic allocation memory)
    extern __shared__ float2 sdata1[];

    int k = (2 * kernel_radius + 1) * (2 * kernel_radius + 1) * trimmed_ratio + 1;

    const int myId = threadIdx.x + threadIdx.y * blockDim.x;

    float2 *max_heap_entry = &sdata1[2 * myId * k];
    float2 *min_heap_entry = (float2 *)&sdata1[(2 * myId + 1) * k];


    int2 left_up = make_int2((x - kernel_radius >= 0) ? (x - kernel_radius) : 0,
                             (y - kernel_radius >= 0) ? (y - kernel_radius) : 0);

    int2 right_bottom = make_int2((x + kernel_radius < width) ? (x + kernel_radius) : width - 1,
                                  (y + kernel_radius < height) ? (y + kernel_radius) : height - 1);

    int cnt = (right_bottom.x - left_up.x + 1) * (right_bottom.y - left_up.y + 1);

    k = cnt * trimmed_ratio;

    // Store the ratio min values and max values
    Max_heap max_heap(k, max_heap_entry);
    Min_heap min_heap(k, min_heap_entry);

    float2 cur_value;
    float k_sum = 0.0f;
    float value_sum = 0.0f;


    for(int r = left_up.y; r <= right_bottom.y; ++r){
        for(int c = left_up.x; c <= right_bottom.x; ++c){
            cur_value.x = tex2D<float>(texObj, float(c) + 0.5, float(r) + 0.5);

            cur_value.y = (1/(2 * MATH_PI * sigma * sigma)) * exp(-((r - y) * (r - y) + (c - x) * (c - x)) / (2 * sigma * sigma));

            value_sum += cur_value.x * cur_value.y;
            k_sum += cur_value.y;

            if(!max_heap.full()){
                max_heap.insert2(cur_value);
                min_heap.insert2(cur_value);
            }else if(cur_value.x < max_heap.top2().x){
                max_heap.delete_and_insert2(cur_value);
            }else if(cur_value.x > min_heap.top2().x){
                min_heap.delete_and_insert2(cur_value);
            }
        }
    }

    // Step 2: Read from texture amd write to global memory
    output[y * width + x] = (value_sum - max_heap.sum2() - min_heap.sum2()) / (k_sum - min_heap.sumy() - max_heap.sumy());
    //printf("\noutput: %f",output[y * width + x]);
    //printf("\nk_sum - max_k_heap.sum() - min_k_heap.sum(): %f",k_sum - max_k_heap.sum() - min_k_heap.sum());
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


void ImageSmoother::image_smooth(float *d_array, int kernel_radius, float trimmed_ratio, int width, int height, float sigma) {
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

    trimmed_gaussian <<<dimGrid, dimBlock, sizeof(float) * heap_size * 4 * dimBlock.x * dimBlock.y >>>
                                        (d_array, texObj, kernel_radius, trimmed_ratio, width, height, sigma);
    // Finally, clean up
    //  1) Destroy texture object
    cudaDestroyTextureObject(texObj);
    //  2) Free device memory
    cudaFreeArray(cuArray);
}