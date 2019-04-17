//
// Created by feng on 19-4-17.
//

#include "SceneConfiguration.h"
#include "SceneLoader.h"
#include "SceneProcessor.h"
#include "TaskLoader.h"
#include "gtest/gtest.h"

class TaskLoaderFixture : public ::testing::Test{
protected:
    void SetUp(){
        taskLoader = new TaskLoader();

        solarScene = SolarScene::GetInstance();
        SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
        sceneConfiguration->loadConfiguration("test_file/example_configuration.json");

        SceneLoader sceneLoader;
        sceneLoader.SceneFileRead(solarScene, "test_file/example_scene.scn");

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



TEST_F(TaskLoaderFixture, loadGoodExample) {
    std::string good_path = "test_file/taskLoad_goodExample.txt";
    EXPECT_TRUE(taskLoader->loadRayTracingHeliostatIndex(good_path, *solarScene));
}

