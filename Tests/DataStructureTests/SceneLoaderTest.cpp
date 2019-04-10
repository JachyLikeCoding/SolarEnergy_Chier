//
// Created by feng on 19-4-10.
//

#include "SceneLoader.h"
#include "gtest/gtest.h"

class SceneLoaderFixture : public ::testing::Test {
public:
    SceneLoaderFixture() : solarScene(nullptr){}
    SceneLoader *sceneLoader;
    SolarScene *solarScene;

protected:
    void SetUp(){
        solarScene = SolarScene::GetInstance();
        sceneLoader = new SceneLoader();
    }

    void TearDown(){
        delete(sceneLoader);
    }
};

TEST_F(SceneLoaderFixture, goodExample) {
    std::string goodExamplePath = "/home/feng/SolarEnergy_Chier/Tests/DataStructureTests/test_file/test_scene_good.scn";
    EXPECT_TRUE(sceneLoader->SceneFileRead(solarScene, goodExamplePath));
}

TEST_F(SceneLoaderFixture, badExampleWithUnknowmFields){
    std::string badExamplePath = "/home/feng/SolarEnergy_Chier/Tests/DataStructureTests/test_file/test_scene_bad_unknownFields.scn";
    EXPECT_FALSE(sceneLoader->SceneFileRead(solarScene, badExamplePath));
}

TEST_F(SceneLoaderFixture, badExampleNonRegularExpressionFormat){
    std::string badExamplePath = "/home/feng/SolarEnergy_Chier/Tests/DataStructureTests/test_file/test_scene_bad_nonRE.scn";
    EXPECT_FALSE(sceneLoader->SceneFileRead(solarScene,badExamplePath));
}

TEST_F(SceneLoaderFixture, badExampleIncorrectNumberOfCorresponding){
    std::string badExamplePath = "/home/feng/SolarEnergy_Chier/Tests/DataStructureTests/test_file/test_scene_bad_corresponding.scn";
    EXPECT_FALSE(sceneLoader->SceneFileRead(solarScene,badExamplePath));

    solarScene->clear();
    badExamplePath = "/home/feng/SolarEnergy_Chier/Tests/DataStructureTests/test_file/test_scene_bad_corresponding2.scn";
    EXPECT_FALSE(sceneLoader->SceneFileRead(solarScene,badExamplePath));
}





















