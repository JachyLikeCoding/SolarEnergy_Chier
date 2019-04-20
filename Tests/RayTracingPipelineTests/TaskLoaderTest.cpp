//
// Created by feng on 19-4-17.
//

#include <SceneConfiguration.h>
#include <SceneLoader.h>
#include <SceneProcessor.h>
#include <string>

#include "TaskLoader.h"
#include "gtest/gtest.h"


class TaskLoaderFixture : public ::testing::Test{
protected:
    void SetUp(){
        taskLoader = new TaskLoader();

        solarScene = SolarScene::GetInstance();
        SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
        sceneConfiguration->loadConfiguration("/home/feng/SolarEnergy_Chier/Tests/RayTracingPipelineTests/test_file/example_configuration.json");

        SceneLoader sceneLoader;
        sceneLoader.SceneFileRead(solarScene, "/home/feng/SolarEnergy_Chier/Tests/RayTracingPipelineTests/test_file/example_scene.scn");

        SceneProcessor sceneProcessor(sceneConfiguration);
        sceneProcessor.processScene(solarScene);
    }

    void TearDown() {
        delete(taskLoader);
        solarScene->clear();
    }


public:
    TaskLoaderFixture():taskLoader(nullptr),solarScene(nullptr){}

    TaskLoader *taskLoader;
    SolarScene *solarScene;
};

TEST_F(TaskLoaderFixture, loadExampleWithInvalidElements) {
    std::string invalid_path1 = "/home/feng/SolarEnergy_Chier/Tests/RayTracingPipelineTests/test_file/taskLoadWithInvalidElements1.txt";
    std::string invalid_path2 = "/home/feng/SolarEnergy_Chier/Tests/RayTracingPipelineTests/test_file/taskLoadWithInvalidElements2.txt";
    std::string invalid_path3 = "/home/feng/SolarEnergy_Chier/Tests/RayTracingPipelineTests/test_file/taskLoadWithInvalidElements3.txt";

    EXPECT_ANY_THROW(taskLoader->loadRayTracingHeliostatIndex(invalid_path1, *solarScene));
    EXPECT_ANY_THROW(taskLoader->loadRayTracingHeliostatIndex(invalid_path2, *solarScene));
    EXPECT_ANY_THROW(taskLoader->loadRayTracingHeliostatIndex(invalid_path3, *solarScene));
}

TEST_F(TaskLoaderFixture, loadGoodExample) {
    std::string good_path = "/home/feng/SolarEnergy_Chier/Tests/RayTracingPipelineTests/test_file/taskLoad_goodExample.txt";
    EXPECT_TRUE(taskLoader->loadRayTracingHeliostatIndex(good_path, *solarScene));
}


TEST_F(TaskLoaderFixture, loadExampleWithLessThanNIndexes){
    std::string lessThanN_path = "/home/feng/SolarEnergy_Chier/Tests/RayTracingPipelineTests/test_file/taskLoadWithLessThanNIndexes.txt";
    EXPECT_ANY_THROW(taskLoader->loadRayTracingHeliostatIndex(lessThanN_path, *solarScene));
}


TEST_F(TaskLoaderFixture, loadNonExistExample) {
    std::string non_exist_path = "/home/feng/SolarEnergy_Chier/Tests/RayTracingPipelineTests/test_file/nonExistFile.txt";
    EXPECT_ANY_THROW(taskLoader->loadRayTracingHeliostatIndex(non_exist_path, *solarScene));
}


TEST_F(TaskLoaderFixture, loadExampleWithRepetitiveElements) {
    std::string repetitiveElements_path = "/home/feng/SolarEnergy_Chier/Tests/RayTracingPipelineTests/test_file/taskLoadWithRepetitiveElements.txt";
    EXPECT_FALSE(taskLoader->loadRayTracingHeliostatIndex(repetitiveElements_path, *solarScene));
}


TEST_F(TaskLoaderFixture, loadExampleWithMoreThanNIndexes) {
    std::string moreThanN_path = "test_file/taskLoadWithMoreThanNIndexes.txt";
    taskLoader->loadRayTracingHeliostatIndex(moreThanN_path, *solarScene);
    EXPECT_FALSE(taskLoader->loadRayTracingHeliostatIndex(moreThanN_path, *solarScene));
}
