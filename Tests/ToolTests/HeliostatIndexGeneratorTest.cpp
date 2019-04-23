//
// Created by feng on 19-4-22.
//

#include <iostream>
#include "gtest/gtest.h"
#include "ImageSaver.h"
#include "GenerateHeliostatIndex.h"


TEST(GenerateHeliostatIndex, generateHeliostatIndexTXT){
    int helio_number = 70123;

    std::string filename = "test_output/" + std::to_string(helio_number) + "_heliostats_index.txt";
    GenerateHeliostatIndexTXT HeliostatIndexTXT;
    HeliostatIndexTXT.generateHeliostatIndexTXT(filename, helio_number);
}