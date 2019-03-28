//
// Created by feng on 19-3-28.
//
#include "RandomGenerator.cuh"
#include "check_cuda.h"]
#include "global_function.cuh"

curandGenerator_t RandomGenerator::gen;

void RandomGenerator::initCudaRandGenerator() {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
}

void RandomGenerator::destroyCudaRandGenerator() {
    gen = nullptr;
    curandDestroyGenerator(gen);
}

__global__ void map_float2int(int *d_intData, float const *d_floatData, int low_threshold, int high_threshold, size_t size){
    unsigned int myId = global_func::getThreadID();
    if(myId >= size) return;

    d_intData[myId] = int(d_floatData[myId] * (high_threshold - low_threshold) + low_threshold);
}


bool RandomGenerator::gpu_Uniform(int *d_min_max_array, int low_threshold, int high_threshold, int array_length) {
    if(d_min_max_array == nullptr){
        return false;
    }

    float *d_uniform = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_uniform,  sizeof(float) * array_length));
    curandGenerateUniform(gen, d_uniform, array_length);

    int nThreads;
    dim3 nBlocks;
    if(!global_func::setThreadBlocks(nBlocks, nThreads, array_length)){
        return false;
    }

    map_float2int << < nBlocks, nThreads > >>(d_min_max_array, d_uniform, low_threshold, high_threshold, array_length);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_uniform));
    return true;
}