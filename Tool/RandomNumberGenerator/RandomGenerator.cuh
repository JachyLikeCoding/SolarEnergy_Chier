//
// Created by feng on 19-3-28.
//
#ifndef SOLARENERGY_CHIER_RANDOMGENERATOR_CUH
#define SOLARENERGY_CHIER_RANDOMGENERATOR_CUH

#include <curand.h>
#include <random>

class RandomGenerator{
public:
    //static unsigned int seed
    static curandGenerator_t gen;
    static std::default_random_engine generator;

    //init the seed
    static void initSeed();

    static void initSeed(unsigned int seed);

    static void initCudaRandGenerator();

    //destroy the cuda random generator
    static void destroyCudaRandGenerator();

    //[0 , 1]
    static bool cpu_Uniform(float *h_0_1_array, int array_length);
    static bool gpu_Uniform(float *d_0_1_array, int array_length);

    static bool cpu_Gaussian(float *h_0_1_array, float mean, float stddev, int array_length);
    static bool gpu_Gaussian(float *d_0_1_array, float mean, float stddev, int array_length);

    //[low_threshold, high_threshold]
    static bool cpu_Uniform(int *h_min_max_array, int low_threshold, int high_threshold, int array_length);
    static bool gpu_Uniform(int *d_min_max_array, int low_threshold, int high_threshold, int array_length);

};


#endif //SOLARENERGY_CHIER_RANDOMGENERATOR_CUH
