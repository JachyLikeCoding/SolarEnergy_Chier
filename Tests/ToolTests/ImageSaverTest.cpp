//
// Created by feng on 19-4-17.
//

#include <chrono>
#include <iostream>
#include "ImageSaver.h"
#include "gtest/gtest.h"

TEST(ImageSaver, saveAllZeros){
    int height = 1000, width = 2000;
    float *h_array = new float[height * width];
    for(int i = 0; i < height * width; ++i){
        h_array[i] = 0.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    ImageSaver::saveText("test_output/all_zeros.txt", height, width, h_array);

    auto end = std::chrono::high_resolution_clock::now();
    long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "\nSave text file for the size of " << height << " * " << width << " large uses " << elapsed << " microseconds.";
}

TEST(ImageSaver, saveWithNonZeroElements) {
    int height = 1000, width = 2000;
    float *h_array = new float[height * width];
    for (int i = 0; i < height * width; ++i) {
        if (i % 50 == 0) {
            h_array[i] = 200.985f;
        } else {
            h_array[i] = 0.0f;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    ImageSaver::saveText("test_output/with_non_zeros.txt", height, width, h_array);
    auto end = std::chrono::high_resolution_clock::now();
    long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "\nSave text file for the size of " << height << " * " << width << " large uses " << elapsed
              << " microseconds.";
}