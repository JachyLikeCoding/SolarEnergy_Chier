//
// Created by feng on 19-4-17.
//

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

#include "ImageSaver.h"
#include "global_constant.h"




void ImageSaver::saveText(std::string filename, int height, int width, float *h_data, int precision, int rows_package){
    std::ofstream fout(filename.c_str());
    std::stringstream ss;

    int address = 0;
    float max_value = 0.0f;
    float summation_value = 0.0f;

    for(int r = 0; r < height; ++r){
        if(r % rows_package == rows_package - 1){
            fout << ss.rdbuf();
            ss.clear();
        }

        for(int c = 0; c < width; ++c){
            address = (height - 1 - r) * width + c;

            if(h_data[address] < Epsilon){
                ss << 0;
            }else{
                max_value = std::max(max_value, h_data[address]);
                summation_value += h_data[address];
                ss << std::fixed << std::setprecision(precision) << h_data[address];
            }

            if(c != width-1){
                ss << ',';
            }else{
                ss << '\n';
            }
        }
    }

    std::cout << "Max value is " << max_value << ". Sum value is " << summation_value << "." << std::endl;

    fout << ss.rdbuf();
    fout.close();

}
