//
// Created by feng on 19-4-22.
//

#include <iostream>
#include <sstream>
#include <fstream>
#include "GenerateHeliostatIndex.h"

void GenerateHeliostatIndexTXT::generateHeliostatIndexTXT(std::string filename, int helio_number){
    std::ofstream fout(filename.c_str());
    std::stringstream ss;

    ss << helio_number;
    ss << "\n";
    for(int i = 0; i < helio_number - 1; ++i){
        ss << i << " ";
    }
    ss << helio_number - 1;

    fout << ss.rdbuf();
    fout.close();
}
