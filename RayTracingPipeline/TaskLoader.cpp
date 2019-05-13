////
//// Created by feng on 19-4-17.
////
//
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <string>
//#include "TaskLoader.h"
//
//bool TaskLoader::loadRayTracingHeliostatIndex(std::string task_path, SolarScene &solarScene) {
//    std::ifstream task_file(task_path);
//    if(task_file.fail()){
//        throw std::runtime_error("Cannot open the file '" + task_path + "'.");
//    }
//
//    std::stringstream task_stream;
//    task_stream << task_file.rdbuf();
//    task_file.close();
//
//    std::string number_buffer;
//    std::string::size_type  character_pos_next_to_num;  //alias of size_t
//    task_stream >> number_buffer;
//    int N;
//
//    // 1. Check whether N is valid
//    try{
//        N = std::stoi(number_buffer, &character_pos_next_to_num);
//    } catch (exception e){
//        throw std::runtime_error("The number of heliostats '" + number_buffer + "' is invalid."
//                                 "\nPlease check the 1st element in the file '" + task_path + ".");
//    }
//
//    if(character_pos_next_to_num < number_buffer.size() || N <= 0){
//        throw std::runtime_error("The number of heliostats '" + number_buffer + "' is invalid."
//                                 "\nPlease check the 1st element in the file '" + task_path + "'.");
//    }
//
//    heliostat_indexes.clear();
//    receiver_indexes.clear();
//    int heliostat_id;
//    bool ans = true;
//
//    for(int i = 0; i < N; ++i){
//        std::cout << "heliostat index = " << i << endl;
//        // No more heliostat indexes
//        if(task_stream.rdbuf()->in_avail() == 0){
//            throw std::runtime_error(std::to_string(N) + " heliostat(s) are expected. But only got " + std::to_string(i) +
//            "heliostats.\nPlease check you file '" + task_path +"'.");
//        }
//
//        task_stream >> number_buffer;
//        // non-numerical character in heliostat indexes
//        try{
//            heliostat_id = std::stoi(number_buffer, &character_pos_next_to_num);
//        } catch (exception e){
//            throw std::runtime_error("The " + std::to_string(i + 1) + "-th heliostat index '" + number_buffer += "' is invalid."
//                                     "\nPlease check your file '" + task_path + "'.");
//        }
//
//        // 1. Invalid heliostat_id :
//        //      1) non-numerical character;
//        //      2) less than 0;
//        //      3) larger than heliostat array size-1
//        if(character_pos_next_to_num < number_buffer.size() ||
//            heliostat_id < 0 ||
//            heliostat_id >= solarScene.getHeliostats().size()){
//            throw std::runtime_error(
//                    "The " + std::to_string(i + 1) + "-th heliostat index '" + number_buffer +=
//                    "' is invalid.\n Please check your file '" + task_path + "'.");
//        }
//
//        if(heliostat_indexes.find(heliostat_id) == heliostat_indexes.end()){
//            heliostat_indexes.insert(heliostat_id);
//            receiver_indexes.insert(solarScene.getHeliostats()[heliostat_id]->getBelongingGridId());
//        }else{  //If repetitive index occurs, print warning messages.
//            std::cout << "Warning: The " + std::to_string(i + 1) + "-th heliostat index '" +
//                         std::to_string(heliostat_id) + "' is repetitive. Please be careful about that." << std::endl;
//            ans = false;
//        }
//
//        // Check whether the number of heliostat indexes is more than N
//        std::string left_part;
//        int n = task_stream.rdbuf()->in_avail();//Gets how many bytes of unprocessed data remain in the cin input buffer.
//        task_stream >> left_part;
//        if(!left_part.empty()){
//            left_part = task_stream.str();
//            left_part = left_part.substr(left_part.size() - n, n);
//            std::cout << "Warning: '" + left_part + "' is not processed since only " + std::to_string(N) +
//            " heliostat(s). Please be careful about that." << std::endl;
//            return false;
//        }
//        return ans;
//    }
//}
//
//
//const std::unordered_set<int> &TaskLoader::getHeliostatIndexesArray() const {
//    return heliostat_indexes;
//}
//
//const std::unordered_set<int> &TaskLoader::getReceiverIndexesArray() const {
//    return receiver_indexes;
//}


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "TaskLoader.h"

bool TaskLoader::loadRayTracingHeliostatIndex(std::string filepath, SolarScene &solarScene) {
    std::ifstream task_file(filepath);
    if (task_file.fail()) {
        throw std::runtime_error("Cannot open the file '" + filepath + "'.");
    }

    std::stringstream task_stream;
    task_stream << task_file.rdbuf();
    task_file.close();

    std::string number_buffer;
    std::string::size_type character_pos_next_to_num;   // alias of size_t
    task_stream >> number_buffer;
    int N;

    // 1. Check the whether N is valid
    try {
        N = std::stoi(number_buffer, &character_pos_next_to_num);
    } catch (exception e) {
        throw std::runtime_error(
                "The number of heliostats '" + number_buffer +
                "' is invalid.\nPlease check the 1st element in the file '" + filepath + "'.");
    }

    if (character_pos_next_to_num < number_buffer.size() || N <= 0) {
        throw std::runtime_error(
                "The number of heliostats '" + number_buffer +
                "' is invalid.\nPlease check the 1st element in the file '" + filepath + "'.");
    }

    heliostat_indexes.clear();
    receiver_indexes.clear();
    int heliostat_id;
    bool ans = true;
    for (int i = 0; i < N; ++i) {

        //std::cout << "heliostat_index = " << i << endl;

        // No more heliostat indexes
        if (task_stream.rdbuf()->in_avail() == 0) {
            throw std::runtime_error(
                    std::to_string(N) + " heliostat(s) are expected. But only got " + std::to_string(i) +
                    " heliostats.\nPlease check your file '" + filepath + "'.");
        }

        task_stream >> number_buffer;
        //  non-numerical character in heliostat index
        try {
            heliostat_id = std::stoi(number_buffer, &character_pos_next_to_num);
        } catch (exception e) {
            throw std::runtime_error(
                    "The " + std::to_string(i + 1) + "-th heliostat index '" + number_buffer +
                    "' is invalid.\nPlease check your file '" + filepath + "'.");
        }

        // 1. Invalid heliostat_id :
        //  1) non-numerical character;
        //  2) less than 0;
        //  3) larger than heliostat array size-1
        if (character_pos_next_to_num < number_buffer.size() ||
            heliostat_id < 0 || heliostat_id >= solarScene.getHeliostats().size()) {
            throw std::runtime_error(
                    "The " + std::to_string(i + 1) + "-th heliostat index '" + number_buffer +
                    "' is invalid.\n Please check your file '" + filepath + "'.");
        }

        if (heliostat_indexes.find(heliostat_id) == heliostat_indexes.end()) {
            heliostat_indexes.insert(heliostat_id);
            receiver_indexes.insert(solarScene.getHeliostats()[heliostat_id]->getBelongingGridId());
        } else {
            // If the repetitive indexes occurs, print warning messages.
            std::cout << "Warning: The " + std::to_string(i + 1) + "-th heliostat index '" +
                         std::to_string(heliostat_id) + "' is repetitive. Please be careful about that." << std::endl;
            ans = false;
        }
    }

    // Check whether the number of heliostat indexes is more than N
    std::string left_part;
    int n = task_stream.rdbuf()->in_avail();
    task_stream >> left_part;
    if (!left_part.empty()) {
        left_part = task_stream.str();
        left_part = left_part.substr(left_part.size() - n, n);
        std::cout << "Warning: '" + left_part + "' is not processed since only " + std::to_string(N) +
                     " heliostat(s). Please be careful about that." << std::endl;
        return false;
    }

    return ans;
}

const std::unordered_set<int> &TaskLoader::getHeliostatIndexesArray() const {
    return heliostat_indexes;
}

const std::unordered_set<int> &TaskLoader::getReceiverIndexesArray() const {
    return receiver_indexes;
}

