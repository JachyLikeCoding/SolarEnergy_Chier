//
// Created by feng on 19-4-10.
//

#include <string>

#include "SceneConfiguration.h"
#include "gtest/gtest.h"

void printSceneConfiguration(SceneConfiguration *sceneConfiguration){
    printf("\nSun ray related fields:\n");
    float3 dir = sceneConfiguration->getSun_dir();
    printf("\tSun Direction - %.4f, %.4f, %.4f\n", dir.x, dir.y, dir.z);
    printf("\tDNI - %.4f\n", sceneConfiguration->getDni());
    printf("\tCSR - %.4f\n", sceneConfiguration->getCsr());
    printf("\tNumber of Sunshape_Groups - %d\n", sceneConfiguration->getNum_sunshape_groups());
    printf("\tNumber of Lights per Sunshape_Group - %d\n", sceneConfiguration->getNum_sunshape_lights_per_group());
    printf("\tNumber of Inverse Transform Sampling Group - %d\n",
           sceneConfiguration->getInverse_transform_sampling_groups());

    printf("\nReceiver related fields:\n");
    printf("\tLength of Receiver Pixel - %.4f\n", sceneConfiguration->getReceiver_pixel_length());

    printf("\nHeliostat related fields:\n");
    printf("\tStandard deviation of Heliostat surface - %.4f\n", sceneConfiguration->getDisturb_std());
    printf("\tLength of Heliostat Pixel - %.4f\n", sceneConfiguration->getHelio_pixel_length());
    printf("\tReflective rate - %.4f\n", sceneConfiguration->getReflected_rate());
}



TEST(loadConfiguration, goodExample) {
    SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
    std::string configuration_path = "/home/feng/SolarEnergy_Chier/Tests/DataStructureTests/test_file/test_configuration.json";
    EXPECT_EQ(sceneConfiguration->num_of_fields, sceneConfiguration->loadConfiguration(configuration_path));

    printf("Good Example:");
    printSceneConfiguration(sceneConfiguration);
}



TEST(loadConfiguration, goodExampleWithPartialSettingFields) {
    SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
    std::string configuration_path = "/home/feng/SolarEnergy_Chier/Tests/DataStructureTests/test_file/test_partial_configuration.json";
    EXPECT_EQ(8, sceneConfiguration->loadConfiguration(configuration_path));

    printf("Good Example with Partial Setting Fields(without sun direction fields):");
    printSceneConfiguration(sceneConfiguration);
}


TEST(loadConfiguration, badExampleWithInvalidJsonFormat){
    SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
    std::string configuration_path = "/home/feng/SolarEnergy_Chier/Tests/DataStructureTests/test_file/test_invalid_configuration.json";
    EXPECT_ANY_THROW(sceneConfiguration->loadConfiguration(configuration_path));

    printf("Bad Example with Invalid Json Format:");
    printSceneConfiguration(sceneConfiguration);
}

TEST(loadConfiguration, badExampleWithNonExistConfigurationFile){
    SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
    std::string configuration_path = "test_file/nonexist_file.json";

    printf("Bad Example With Non-exist Configuration File:");
    EXPECT_ANY_THROW(sceneConfiguration->loadConfiguration(configuration_path));
}