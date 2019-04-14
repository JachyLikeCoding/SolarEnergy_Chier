//
// Created by feng on 19-4-10.
//

#include <iostream>
#include "RandomGenerator.cuh"
#include "gtest/gtest.h"
#include "global_function.cuh"

TEST(RandomGeneratorFunction, initGPUSeed){
    RandomGenerator::initCudaRandGenerator();
}

class RandomGeneratorGPUFixture : public ::testing:: Test{
public:
    float *d_0_1_array;
    int *d_min_max_array;
    int array_size;
    RandomGeneratorGPUFixture() : d_0_1_array(nullptr), d_min_max_array(nullptr), array_size(10){}

    template <typename T>
    void print_devide_array(T *d_array, int array_size){
        // Allocate host array
        T *h_array = new T[array_size];
        global_func::gpu2cpu(h_array, d_array, array_size);

        for(int i = 0; i < array_size; ++i){
            std::cout << "  " << h_array[i];
        }
        std::cout << std::endl;

        //Free host array
        delete[] h_array;
        h_array = nullptr;
    }



protected:
    void SetUp(){
        RandomGenerator::initCudaRandGenerator();
        checkCudaErrors(cudaMalloc((void **) &d_0_1_array, sizeof(float) * array_size));
        checkCudaErrors(cudaMalloc((void **) &d_min_max_array, sizeof(float) * array_size));
    }

    void TearDown(){
        checkCudaErrors(cudaFree(d_min_max_array));
        d_min_max_array = nullptr;
        checkCudaErrors(cudaFree(d_0_1_array));
        d_0_1_array = nullptr;
        RandomGenerator::destroyCudaRandGenerator();
    }
};

TEST_F(RandomGeneratorGPUFixture, Uniform_float){
    std::cout << "\nGPU Uniform (float between [0,1]) : " << std::endl;
    std::cout << "\t- Round 1:";
    RandomGenerator::gpu_Uniform(d_0_1_array, array_size);
    print_devide_array(d_0_1_array, array_size);
    std::cout << "\t- Round 2:";
    RandomGenerator::gpu_Uniform(d_0_1_array, array_size);
    print_devide_array(d_0_1_array, array_size);

    // Print Note message
    std::clog << "\nNoted: The results of two runs should not be the same." << std::endl;
}


TEST_F(RandomGeneratorGPUFixture, Gaussian_float) {
    float mean = 0.0f;
    float stddev = 1.0f;

    std::cout << "\nGPU Gaussian:" << std::endl;

    std::cout << "\t- Round 1:";
    RandomGenerator::gpu_Gaussian(d_0_1_array, mean, stddev, array_size);
    print_devide_array(d_0_1_array, array_size);

    std::cout << "\t- Round 2:";
    RandomGenerator::gpu_Gaussian(d_0_1_array, mean, stddev, array_size);
    print_devide_array(d_0_1_array, array_size);

    // Print Note message
    std::clog << "\nNoted: The results of two runs should not be the same." << std::endl;
}


TEST_F(RandomGeneratorGPUFixture, Uniform_integer) {
    int low_threshold = 0;
    int high_threshold = 10;

    std::cout << "\nGPU Uniform(integer between [0,10)):" << std::endl;

    std::cout << "\t- Round 1:";
    RandomGenerator::gpu_Uniform(d_min_max_array, low_threshold, high_threshold, array_size);
    print_devide_array(d_min_max_array, array_size);

    std::cout << "\t- Round 2:";
    RandomGenerator::gpu_Uniform(d_min_max_array, low_threshold, high_threshold, array_size);
    print_devide_array(d_min_max_array, array_size);

    // Print Note message
    std::clog << "\nNoted: The results of two runs should not be the same." << std::endl;
}
