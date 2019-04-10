//
// Created by feng on 19-4-9.
//

#include <cstring>
#include <stdexcept>
#include <fstream>
#include <sys/stat.h>
#include <getopt.h>

#include "ArgumentParser.h"


void ArgumentParser::initialize() {
    configuration_path = "/home/feng/SolarEnergy_Chier/InputFiles/example/example_configuration.json";
    scene_path = "/home/feng/SolarEnergy_Chier/InputFiles/example/example_scene.scn";
    output_path = "/home/feng/SolarEnergy_Chier/OutputFiles/";
    heliostat_index_load_path = "/home/feng/SolarEnergy_Chier/InputFiles/example/tast_heliostats_index.txt";
}

void ArgumentParser::check_valid_file(std::string file_path, std::string suffix) {
    // 1. Whether  the suffix is at the end of the file path.
    if(file_path.size() < suffix.size() ||
        file_path.substr(file_path.size() - suffix.size(), suffix.size()) != suffix){
        throw std::runtime_error("The path of file ' " + file_path + " ' does not consist of ' " + suffix + " ' suffix.\n");
    }

    // 2. Whether the file path exists
    std::ifstream f(file_path.c_str());
    if(!f.good()){
        throw std::runtime_error("The file couldn't be opened. Please check your file path.\n");
    }
}



void ArgumentParser::check_valid_directory(std::string dir_path) {
    if(dir_path[dir_path.size() - 1] != '/'){
        throw std::runtime_error("' " + dir_path + " ' directory format is incorrect. Please add '/' at the end of the path.\n");
    }
    struct stat info;
    stat(dir_path.c_str(), &info);
    if(!info.st_mode || !S_IFDIR){
        throw std::runtime_error("' " + dir_path + " ' directory does not exist. Please check the path.\n");
    }
}


bool ArgumentParser::parser(int argc, char **argv){
    initialize();

    static struct option long_options[] = {
            {"configuration_path",      required_argument, 0, 'c'},
            {"scene_path",              required_argument, 0, 's'},
            {"output_path",             required_argument, 0, 'o'},
            {"heliostat_index_path",    required_argument, 0, 'h'},
            {0,                         0,                 0,  0 }
    };

    int option_index = 0;
    int c;
    optind = 1; // 'optind' is the index of next element to be processed
                // The caller should set it as 1 to restart scanning of the next argv
    while((c = getopt_long(argc, argv, "c:s:o:h", long_options, &option_index)) != -1){
        if(c == '?'){
            // unknown arguments
            return false;
        }

        if(c == 'c'){
            configuration_path = optarg;
            printf("\noption -c/configuration_path with value '%s'", optarg);
        }else if(c == 's'){
            scene_path = optarg;
            printf("\noption -s/scene_path with value '%s'", optarg);
        }else if(c == 'o'){
            output_path = optarg;
            printf("\noption -o/output_path with value '%s'", optarg);
        }else if(c == 'h'){
            heliostat_index_load_path = optarg;
            printf("\noption -h/heliostat_index_path with value '%s'", optarg);
        }
    }

    check_valid_file(configuration_path, ".json");
    check_valid_file(scene_path, ".scn");
    check_valid_file(heliostat_index_load_path, ".txt");
    check_valid_directory(output_path);

    return true;
}


const std::string &ArgumentParser::getConfigurationPath() const {
    return configuration_path;
}

bool ArgumentParser::setConfigurationPath(const std::string &configuration_path) {
    check_valid_file(configuration_path, ".json");
    ArgumentParser::configuration_path = configuration_path;
    return true;
}

const std::string &ArgumentParser::getScenePath() const {
    return scene_path;
}

bool ArgumentParser::setScenePath(const std::string &scene_path) {
    check_valid_file(scene_path, ".scn");
    ArgumentParser::scene_path = scene_path;
    return true;
}

const std::string &ArgumentParser::getHeliostatIndexLoadPath() const {
    return heliostat_index_load_path;
}

void ArgumentParser::setHeliostatIndexLoadPath(const std::string &heliostat_index_load_path) {
    ArgumentParser::heliostat_index_load_path = heliostat_index_load_path;
}


const std::string &ArgumentParser::getOutputPath() const {
    return output_path;
}

void ArgumentParser::setOutputPath(const std::string &output_path) {
    ArgumentParser::output_path = output_path;
}








