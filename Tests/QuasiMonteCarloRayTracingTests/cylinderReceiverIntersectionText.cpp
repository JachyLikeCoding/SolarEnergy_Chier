//
// Created by feng on 19-4-21.
//

#include <SceneLoader.h>
#include <SceneProcessor.h>
#include <CylinderReceiver.cuh>
#include <RectangleGrid.cuh>

#include "Receiver/cylinderReceiverIntersection.cuh"
#include "QuasiMonteCarloRayTracer.h"
#include "CylinderReceiverRectangleGridRayTracing.cuh"
#include "gtest/gtest.h"
#include "receiverIntersectUtil.h"

class cylinderReceiverIntersectionFixture : public ::testing::Test {
protected:
    void loadAndProcessScene() {
        solarScene = SolarScene::GetInstance();

        SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
        std::string configuration_path = "test_file/test_configuration2.json";
        sceneConfiguration->loadConfiguration(configuration_path);

        std::string scene_path = "test_file/test_scene_cylinder_receiver.scn";
        SceneLoader sceneLoader;
        sceneLoader.SceneFileRead(solarScene, scene_path);

        SceneProcessor sceneProcessor(sceneConfiguration);
        sceneProcessor.processScene(solarScene);
    }

    void SetUp() {
        loadAndProcessScene();
    }

    void TearDown() {
        solarScene->clear();
    }

public:
    void print(vector<float> &array, int2 resolution) {
        for (int r = resolution.y - 1; r >= 0; --r) {
            for (int c = 0; c < resolution.x; ++c) {
                std::cout << array[r * resolution.x + c] << ' ';
            }
            std::cout << std::endl;
        }
    }

    cylinderReceiverIntersectionFixture() : solarScene(nullptr) {}

    QuasiMonteCarloRayTracer QMCRTracer;
    SolarScene *solarScene;
};

TEST_F(cylinderReceiverIntersectionFixture, cylinderReceiverIntersectionParallel) {
    // Change lights to parallel direction
    changeSunLightAndPerturbationToParallel(solarScene->getSunray());

    // Construct arguments
    SunrayArgument sunrayArgument = QMCRTracer.generateSunrayArgument(solarScene->getSunray());
    CylinderReceiver *cylinderReceiver = dynamic_cast<CylinderReceiver *>(solarScene->getReceivers()[0]);
    RectangleGrid *rectGrid = dynamic_cast<RectangleGrid *>(solarScene->getGrids()[0]);
    float factor = 1.0f;
    float3 *d_subHeliostat_vertexes = nullptr;
    int start_heliostat_id = rectGrid->getStartHeliostatIndex();
    int end_heliostat_id = start_heliostat_id + rectGrid->getNumberOfHeliostats();
    QMCRTracer.setFlatRectangleHeliostatVertexs(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                                 start_heliostat_id, end_heliostat_id);

    // Heliostat 1 without shadow(index = 0)
    HeliostatArgument heliostatArgument0 = QMCRTracer.generateHeliostatArgument(solarScene, 0);
    CylinderReceiverRectangleGridRayTracing(sunrayArgument, cylinderReceiver, rectGrid, heliostatArgument0,
                                       d_subHeliostat_vertexes, factor);

    int2 resolution = cylinderReceiver->getResolution();
    vector<float> image = deviceArray2Vector(cylinderReceiver->getDeviceImage(), resolution.y * resolution.x);
    std::cout<<"\nHeliostat 1 without shadowing - (r: "<<resolution.y<<", c:"<<resolution.x<<")"<<std::endl;
    print(image, resolution);

    // Heliostat 1 with shadow(index = 1)
    rectGrid = dynamic_cast<RectangleGrid *>(solarScene->getGrids()[1]);
    start_heliostat_id = rectGrid->getStartHeliostatIndex();
    end_heliostat_id = start_heliostat_id + rectGrid->getNumberOfHeliostats();
    QMCRTracer.setFlatRectangleHeliostatVertexs(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                                 start_heliostat_id, end_heliostat_id);

    HeliostatArgument heliostatArgument1 = QMCRTracer.generateHeliostatArgument(solarScene, 1);
    cylinderReceiver->Cclean_image_content();
    CylinderReceiverRectangleGridRayTracing(sunrayArgument, cylinderReceiver, rectGrid, heliostatArgument1,
                                       d_subHeliostat_vertexes, factor);

    image = deviceArray2Vector(cylinderReceiver->getDeviceImage(), resolution.y * resolution.x);
    std::cout<<"\nHeliostat 1 with shadowing - (r: "<<resolution.y<<", c:"<<resolution.x<<")"<<std::endl;
    print(image, resolution);
}

TEST_F(cylinderReceiverIntersectionFixture, cylinderReceiverIntersectionForRealGrid3) {
    // Construct arguments
    SunrayArgument sunrayArgument = QMCRTracer.generateSunrayArgument(solarScene->getSunray());
    CylinderReceiver *cylinderReceiver = dynamic_cast<CylinderReceiver *>(solarScene->getReceivers()[0]);
    RectangleGrid *rectGrid = dynamic_cast<RectangleGrid *>(solarScene->getGrids()[2]);
    float factor = 1.0f;
    float3 *d_subHeliostat_vertexes = nullptr;
    int start_heliostat_id = rectGrid->getStartHeliostatIndex();
    int end_heliostat_id = start_heliostat_id + rectGrid->getNumberOfHeliostats();
    QMCRTracer.setFlatRectangleHeliostatVertexs(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                                 start_heliostat_id, end_heliostat_id);

    // Heliostat 4 without shadow(index = 3)
    HeliostatArgument heliostatArgument0 = QMCRTracer.generateHeliostatArgument(solarScene, 3);
    CylinderReceiverRectangleGridRayTracing(sunrayArgument, cylinderReceiver, rectGrid, heliostatArgument0,
                                       d_subHeliostat_vertexes, factor);

    int2 resolution = cylinderReceiver->getResolution();
    vector<float> image = deviceArray2Vector(cylinderReceiver->getDeviceImage(), resolution.y * resolution.x);
    std::cout<<"\nHeliostat 1 without shadowing - (r: "<<resolution.y<<", c:"<<resolution.x<<")"<<std::endl;
    print(image, resolution);
}