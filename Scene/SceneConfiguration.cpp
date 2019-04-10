//
// Created by feng on 19-3-31.
//
#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"
#include "SceneConfiguration.h"

SceneConfiguration *SceneConfiguration::sceneConfigurationInstance;

SceneConfiguration *SceneConfiguration::getInstance() {
    if(sceneConfigurationInstance == nullptr){
        sceneConfigurationInstance = new SceneConfiguration();
    }
    return sceneConfigurationInstance;
}

template <typename T>
int set_field(T &field, std::string field_json_format, nlohmann::json &json){
    if(json.find(field_json_format) != json.end()){
        field = json[field_json_format];
        return 1;
    }
    return 0;
}

template <typename T>
int set_3field(T &field, std::string field_json_format, nlohmann::json &json){
    if(json.find(field_json_format) != json.end()){
        field.x = json[field_json_format][0];
        field.y = json[field_json_format][1];
        field.z = json[field_json_format][2];
        return 1;
    }
    return 0;
}

int SceneConfiguration::loadConfiguration(std::string configuration_file_path) {
    std::ifstream inputJsonFile(configuration_file_path);
    if(inputJsonFile.fail()){
        std::string error_message = "The file '" + configuration_file_path + "' doesn't exist";
        std::cerr << error_message << std::endl;
        throw std::runtime_error(error_message);
    }
    nlohmann::json json;

    /**
     * TODO: if the json file is invalid, please deal with the exception.
     */
     inputJsonFile >> json;

     int ans = 0;
     ans += set_3field(sun_dir, "sun_dir", json);
     ans += set_field(dni, "dni", json);
     ans += set_field(csr, "csr", json);
     ans += set_field(num_sunshape_groups, "num_of_sunshape_groups", json);
     ans += set_field(num_sunshape_lights_per_group, "num_per_sunshape_group", json);
     ans += set_field(inverse_transform_sampling_groups, "inverse_transform_sampling_groups", json);
     ans += set_field(receiver_pixel_length, "receiver_pixel_length", json);
     ans += set_field(disturb_std, "helio_disturb_std", json);
     ans += set_field(helio_pixel_length, "helio_pixel_length", json);
     ans += set_field(reflected_rate, "helio_reflected_rate", json);

     if(ans != num_of_fields){
         printf("Noted: not all fields are set. You may need to check your configuration file.\n");
     }
     return ans;
}

const float3 &SceneConfiguration::getSun_dir() const {
    return sun_dir;
}

float SceneConfiguration::getDni() const {
    return dni;
}

float SceneConfiguration::getCsr() const {
    return csr;
}

int SceneConfiguration::getNum_sunshape_groups() const {
    return num_sunshape_groups;
}

int SceneConfiguration::getNum_sunshape_lights_per_group() const {
    return num_sunshape_lights_per_group;
}

float SceneConfiguration::getReceiver_pixel_length() const {
    return receiver_pixel_length;
}

float SceneConfiguration::getDisturb_std() const {
    return disturb_std;
}

float SceneConfiguration::getHelio_pixel_length() const {
    return helio_pixel_length;
}

float SceneConfiguration::getReflected_rate() const {
    return reflected_rate;
}

int SceneConfiguration::getInverse_transform_sampling_groups() const {
    return inverse_transform_sampling_groups;
}






























