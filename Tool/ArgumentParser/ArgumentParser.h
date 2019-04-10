//
// Created by feng on 19-4-9.
//

#ifndef SOLARENERGY_CHIER_ARGUMENTPARSER_H
#define SOLARENERGY_CHIER_ARGUMENTPARSER_H


#include <string>
class ArgumentParser{
private:
    std::string configuration_path;
    std::string scene_path;
    std::string heliostat_index_load_path;
    std::string output_path;

    void initialize();
    void check_valid_file(std::string path, std::string suffix);
    void check_valid_directory(std::string path);


public:
    bool parser(int argc, char **argv);

    const std::string &getConfigurationPath() const;
    bool setConfigurationPath(const std::string &configuration_path);

    const std::string &getScenePath() const;
    bool setScenePath(const std::string &scene_path);

    const std::string &getHeliostatIndexLoadPath() const;
    void setHeliostatIndexLoadPath(const std::string &heliostat_index_load_path);

    const std::string &getOutputPath() const;
    void setOutputPath(const std::string &output_path);

};


#endif //SOLARENERGY_CHIER_ARGUMENTPARSER_H
