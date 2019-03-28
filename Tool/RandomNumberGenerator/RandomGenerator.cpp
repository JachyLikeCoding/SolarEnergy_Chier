//
// Created by feng on 19-3-28.
//
#include "check_cuda.h"
#include "RandomGenerator.cuh"

std::default_random_engine RandomGenerator::generator;

void RandomGenerator::initSeed() {
    unsigned int mrand = (unsigned int) time(NULL);
    srand(mrand);
    generator.seed(mrand);
}


void RandomGenerator::initSeed(unsigned int seed) {
    srand(seed);
    generator.seed(seed);
}

bool RandomGenerator::cpu_Uniform(float *h_0_1_array, int array_length) {
    if(h_0_1_array == nullptr){
        return false;
    }

    for(int i = 0; i < array_length; ++i){
        h_0_1_array[i] = (float)((float)rand() / RAND_MAX);
    }
    return true;
}

bool RandomGenerator::gpu_Uniform(float *d_0_1_array, int array_length) {
    if(d_0_1_array == nullptr){
        return false;
    }
    //Generate 0-1 array
   curandGenerateUniform(gen, d_0_1_array, array_length);
    return true;
}


bool RandomGenerator::cpu_Gaussian(float *h_0_1_array, float mean, float stddev, int array_length) {
    if(h_0_1_array == nullptr){
        return false;
    }
    std::normal_distribution<float> distribution(mean, stddev);

    for(int i = 0; i < array_length; ++i){
        h_0_1_array[i] = distribution(generator);
    }
    return true;
}

bool RandomGenerator::gpu_Gaussian(float *d_0_1_array, float mean, float stddev, int array_length) {
    if(d_0_1_array == nullptr){
        return false;
    }

    curandGenerateNormal(gen, d_0_1_array, array_length, mean, stddev);
    return true;
}

bool RandomGenerator::cpu_Uniform(int *h_min_max_array, int low_threshold, int high_threshold, int array_length) {
    if(h_min_max_array == nullptr){
        return false;
    }

    int range = high_threshold - low_threshold;
    for(int i = 0; i < array_length; ++i){
        h_min_max_array[i] = int(float(rand()) / float(RAND_MAX) * range + low_threshold);
    }
    return true;
}
