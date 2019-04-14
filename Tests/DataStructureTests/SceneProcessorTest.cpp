//
// Created by feng on 19-4-14.
//

#include "gtest/gtest.h"
#include "SceneProcessor.h"
#include "global_function.cuh"
#include "RandomNumberGenerator/RandomGenerator.cuh"



class SceneProcessorFixture : public ::testing::Test {
public:
    SceneProcessor *sceneProcessor;
    SceneConfiguration *sceneConfiguration;

    SceneProcessorFixture() : sceneProcessor(nullptr), sceneConfiguration(nullptr){}

    void print(std::string name, float3 *d_array, int n){
        float3 *h_array = nullptr;
        global_func::gpu2cpu(h_array, d_array, n);

        std::cout << "\nPrint " << name << std::endl;
        for(int i = 0; i < n; ++i){
            std::cout << "(" << h_array[i].x << ", " << h_array[i].y << ", " << h_array[i].z << ")" << std::endl;
        }
        delete[] h_array;
        h_array = nullptr;
    }

protected:
    void SetUp(){
        SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
        std::string configuration_path = "/home/feng/SolarEnergy_Chier/Tests/DataStructureTests/test_file/test_configuration.json";
        sceneConfiguration->loadConfiguration(configuration_path);

        sceneProcessor = new SceneProcessor(sceneConfiguration);

        // test needs random numbers
        RandomGenerator::initCudaRandGenerator();
    }

    void TearDown(){
        delete(sceneProcessor);
        sceneProcessor = nullptr;
        RandomGenerator::destroyCudaRandGenerator();
    }

};


TEST_F(SceneProcessorFixture, set_sunray_content){
    Sunray sunray;
    sceneProcessor->set_sunray_content(sunray);
    int print_size = 10;
    print("Perturbation", sunray.getDevicePerturbation(), min(print_size, sunray.getNumOfSunshapeLightsPerGroup()));
    print("SampleLights", sunray.getDeviceSampleLights(), min(print_size, sunray.getNumOfSunshapeLightsPerGroup()));
}



TEST_F(SceneProcessorFixture, set_grid_receiver_heliostat_content){
    std::vector<Grid *> grids;
    std::vector<Receiver *> receivers;
    std::vector<Heliostat *> heliostats;
    sceneProcessor->set_grid_receiver_heliostat_content(grids, receivers, heliostats);

}
