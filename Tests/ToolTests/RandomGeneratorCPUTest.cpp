//
// Created by feng on 19-4-10.
//
#include <iostream>

#include "RandomGenerator.cuh"
#include "gtest/gtest.h"

TEST(RandomGeneratorFunction, initCPUSeed){
    RandomGenerator::initSeed();
}

class RandomGeneratorCPUFixture : public ::testing::Test {
public:
    float *h_0_1_array;
    int *h_min_max_array;
    int array_size;
    RandomGeneratorCPUFixture() : h_0_1_array(nullptr), h_min_max_array(nullptr), array_size(10){}

    template <typename T>
    void print_host_array(T *array, int array_size){
        for(int i = 0; i < array_size; ++i){
            std::cout << "  " << array[i];
        }
        std::cout << "  " << std::endl;
    }


protected:
    void SetUp(){
        RandomGenerator::initSeed();
        h_0_1_array = new float[array_size];
        h_min_max_array = new int[array_size];
    }

    void TearDown(){
        delete[] h_0_1_array;
        h_0_1_array = nullptr;

        delete[] h_min_max_array;
        h_min_max_array = nullptr;
    }
};


TEST_F(RandomGeneratorCPUFixture, Uniform_float){
    std::cout << "CPU Uniform (float between [0 , 1]): " << std::endl;

    std::cout << "\t- Round 1:";
    RandomGenerator::cpu_Uniform(h_0_1_array, array_size);
    print_host_array(h_0_1_array, array_size);

    std::cout << "\t- Round 2:";
    RandomGenerator::cpu_Uniform(h_0_1_array, array_size);
    print_host_array(h_0_1_array, array_size);

    //Print Note message
    std::clog << "Noted: The results of two runs should not be the same." << std::endl;
}

TEST_F(RandomGeneratorCPUFixture, Gaussian_float){
    float mean = 0.0f;
    float stddev = 1.0f;

    std::cout << "\nCPU Gaussian:" << std::endl;

    std::cout << "\t- Round 1:";
    RandomGenerator::cpu_Gaussian(h_0_1_array, mean, stddev, array_size);
    print_host_array(h_0_1_array, array_size);

    std::cout << "\t- Round 2:";
    RandomGenerator::cpu_Gaussian(h_0_1_array, mean, stddev, array_size);
    print_host_array(h_0_1_array, array_size);

    //Print Note message
    std::clog << "Noted: The results of two runs should not be the same." << std::endl;
}


TEST_F(RandomGeneratorCPUFixture, Uniform_integer) {
    int low_threshold = 0;
    int high_threshold = 10;

    std::cout << "CPU Uniform(float between [1,10])" << std::endl;

    std::cout << "\t- Round 1:";
    RandomGenerator::cpu_Uniform(h_min_max_array, low_threshold, high_threshold, array_size);
    print_host_array(h_min_max_array, array_size);

    std::cout << "\t- Round 2:";
    RandomGenerator::cpu_Uniform(h_min_max_array, low_threshold, high_threshold, array_size);
    print_host_array(h_min_max_array, array_size);

    // Print Note message
    std::clog << "Noted: The results of two runs should not be the same." << std::endl;
}
