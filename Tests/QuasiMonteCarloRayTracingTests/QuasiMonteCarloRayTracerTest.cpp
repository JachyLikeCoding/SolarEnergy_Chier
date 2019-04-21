//
// Created by feng on 19-4-21.
//

//
// Created by dxt on 18-11-21.
//

#include <SceneLoader.h>
#include <SceneProcessor.h>
#include <RectangleGrid.cuh>

#include "NewTypeHeliostatForTest.h"
#include "QuasiMonteCarloRayTracer.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

class QMCRTracerTestFixture : public ::testing::Test {
protected:
    void SetUp() {
        solarScene = SolarScene::GetInstance();

        std::string configuration_path = "test_file/test_configuration.json";

        SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
        sceneConfiguration->loadConfiguration(configuration_path);

        std::string scene_path = "test_file/test_scene.scn";
        SceneLoader sceneLoader;
        sceneLoader.SceneFileRead(solarScene, scene_path);

        Heliostat *h2 = new NewTypeHeliostatForTest();
        h2->setPosition(make_float3(0.0f, 0.0f, 12.0f));
        h2->setSize(make_float3(5.0f, 0.1f, 4.0f));
        h2->setRowAndColumn(make_int2(3, 2));
        h2->setGap(make_float2(1.0f, 0.5f));

        Heliostat *h_tmp = solarScene->getHeliostats()[1];
        solarScene->getHeliostats()[1] = h2;
        delete (h_tmp);

        /**
         * TODO: For unknown reasons, when run the tests from class entrance, the second case will fail.
         * The failure is caused by nullptr sunray in solarScene. But the reason why sunray is nullptr, I just do
         * not understand.
         *
         * Thus, run the tests in this file one by one.
         * */
        SceneProcessor sceneProcessor(sceneConfiguration);
        sceneProcessor.processScene(solarScene);
    }

    void TearDown() {
        solarScene->clear();
    }

public:
    std::vector<float3> deviceArray2vector(float3 *d_array, int size) {
        vector<float3> ans;
        float3 *h_array = nullptr;

        global_func::gpu2cpu(h_array, d_array, size);
        for (int i = 0; i < size; ++i) {
            ans.push_back(h_array[i]);
        }

        delete[] h_array;
        h_array = nullptr;
        return ans;
    }

    QMCRTracerTestFixture() : solarScene(nullptr) {}

    QuasiMonteCarloRayTracer QMCRTracer;

    SolarScene *solarScene;
};

bool Float3Eq(float3 n1, float3 n2, float gap) {
    return (n1.x > n2.x - gap && n1.x < n2.x + gap) &&
           (n1.y > n2.y - gap && n1.y < n2.y + gap) &&
           (n1.z > n2.z - gap && n1.z < n2.z + gap);
}

MATCHER_P(FloatNearPointwise, gap, "Check whether two float3 objects are almost the same") {
    float3 n1 = std::get<0>(arg);
    float3 n2 = std::get<1>(arg);

    *result_listener << "\nExpect value: (" << n1.x << ", " << n1.y << ", " << n1.z << ")\n";
    *result_listener << "Got value: (" << n2.x << ", " << n2.y << ", " << n2.z << ")";
    return Float3Eq(n1, n2, (float) gap);
}

TEST_F(QMCRTracerTestFixture, setFlatRectangleHeliostatVertexes) {
    float3 *d_heliostat_vertexes = nullptr;
    Grid *rectGrid = solarScene->getGrids()[0];
    int start_id = rectGrid->getStartHeliostatIndex();
    cout << "start_id: " << start_id << endl;
    int end_id = start_id + rectGrid->getNumberOfHeliostats();
    cout << "end_id: " << end_id << endl;
    int heliostatVertexesSize = QMCRTracer.setFlatRectangleHeliostatVertexs(d_heliostat_vertexes,
                                                                             solarScene->getHeliostats(), start_id,
                                                                             end_id);
    cout << "heliostatVertexesSize: " << heliostatVertexesSize << endl;
    std::vector<float3> subHelioVertexes = deviceArray2vector(d_heliostat_vertexes, heliostatVertexesSize);

    for(float3 v : subHelioVertexes){
        cout << v.x << ", " << v.y << " , " << v.z << endl;
    }
    std::vector<float3> expectSubHelioVertexes = std::vector<float3>({
         // heliostat1
         make_float3(2.0f, 1.5f, 7.85f), make_float3(2.0f, -1.5f, 7.85f), make_float3(-2.0f, -1.5f, 7.85f),
         // heliostat2
         make_float3(2.5f, -1.0f, 11.95f), make_float3(2.5f, -2.0f, 11.95f), make_float3(0.5f, -2.0f, 11.95f),
         make_float3(-0.5f, -1.0f, 11.95f), make_float3(-0.5f, -2.0f, 11.95f), make_float3(-2.5f, -2.0f, 11.95f),
         make_float3(2.5f, 0.5f, 11.95f), make_float3(2.5f, -0.5f, 11.95f), make_float3(0.5f, -0.5f, 11.95f),
         make_float3(-0.5f, 0.5f, 11.95f), make_float3(-0.5f, -0.5f, 11.95f), make_float3(-2.5f, -0.5f, 11.95f),
         make_float3(2.5f, 2.0f, 11.95f), make_float3(2.5f, 1.0f, 11.95f), make_float3(0.5f, 1.0f, 11.95f),
         make_float3(-0.5f, 2.0f, 11.95f), make_float3(-0.5f, 1.0f, 11.95f), make_float3(-2.5f, 1.0f, 11.95f),
         // heliostat3
         make_float3(-3.40423f, 2.09987f, 20.1253f), make_float3(-3.40541f, -1.90012f, 20.135f),
         make_float3(-6.38378f, -1.90012f,19.7754f)});

    EXPECT_THAT(expectSubHelioVertexes, ::testing::Pointwise(FloatNearPointwise(1e-2), subHelioVertexes));
}

TEST_F(QMCRTracerTestFixture, generateHeliostatArgument) {
    HeliostatArgument heliostatArgument = QMCRTracer.generateHeliostatArgument(solarScene, 0);

    std::vector<float3> microHelioOrigins = deviceArray2vector(heliostatArgument.d_microHelio_origins, heliostatArgument.numberOfMicroHeliostats);
    std::vector<float3> microHelioNormals = deviceArray2vector(heliostatArgument.d_microHelio_normals, heliostatArgument.numberOfMicroHeliostats);

    // 1. Check normals
    std::vector<float3> expectNormals(heliostatArgument.numberOfMicroHeliostats, make_float3(0.0f, 0.0f, -1.0f));
    EXPECT_THAT(expectNormals, ::testing::Pointwise(FloatNearPointwise(1e-3), microHelioNormals));

    // 2. Check micro-heliostat centers
    std::vector<float3> expectOrigins({make_float3(1.5f, 1.0f, 7.85f), make_float3(0.5f, 1.0f, 7.85f),
                                       make_float3(-0.5f, 1.0f, 7.85f), make_float3(-1.5f, 1.0f, 7.85f),
                                       make_float3(1.5f, 0.0f, 7.85f), make_float3(0.5f, 0.0f, 7.85f),
                                       make_float3(-0.5f, 0.0f, 7.85f), make_float3(-1.5f, 0.0f, 7.85f),
                                       make_float3(1.5f, -1.0f, 7.85f), make_float3(0.5f, -1.0f, 7.85f),
                                       make_float3(-0.5f, -1.0f, 7.85f), make_float3(-1.5f, -1.0f, 7.85f)});
    EXPECT_THAT(expectOrigins, ::testing::Pointwise(FloatNearPointwise(1e-3), microHelioOrigins));

    // 3. Check the random starting positions
    int *h_microhelio_belonging_groups = nullptr;
    global_func::gpu2cpu(h_microhelio_belonging_groups, heliostatArgument.d_microHelio_groups,
                         heliostatArgument.numberOfMicroHeliostats);

    std::cout << "\nThe starting position of each micro-heliostats are following:" << std::endl;
    for (int i = 0; i < heliostatArgument.numberOfMicroHeliostats; ++i) {
        std::cout << h_microhelio_belonging_groups[i] << " ";
    }

    delete[] h_microhelio_belonging_groups;
    h_microhelio_belonging_groups = nullptr;

    // 4. Check the subHeliostat index of heliostatArgument
    //    Since the second heliostat composed by 2*3 sub-heliostats, the third heliostat should begin from 7( = 1 + 2*3 )
    HeliostatArgument heliostatArgument2 = QMCRTracer.generateHeliostatArgument(solarScene, 2);
    EXPECT_EQ(heliostatArgument2.subHeliostat_id, 7);

    // 5. Clean up.
    heliostatArgument.CClear();
}


TEST_F(QMCRTracerTestFixture, generateSunrayArgument) {
    SunrayArgument sunrayArgument = QMCRTracer.generateSunrayArgument(solarScene->getSunray());

    // 1. Check sample lights
    std::vector<float3> sunray_samplelights = deviceArray2vector(sunrayArgument.d_samplelights,
                                                                 min(10, sunrayArgument.pool_size));
    std::cout << "The sunray samplelights directions\n";
    for (auto it:sunray_samplelights) {
        std::cout << "(" << it.x << ", " << it.y << ", " << it.z << ")" << std::endl;
    }

    // 2. Check perturbation lights
    std::vector<float3> sunray_perturbations = deviceArray2vector(sunrayArgument.d_perturbations,
                                                                  min(10, sunrayArgument.pool_size));
    std::cout << "\nThe sunray perturbations directions\n";
    for (auto it:sunray_perturbations) {
        std::cout << "(" << it.x << ", " << it.y << ", " << it.z << ")" << std::endl;
    }
}

TEST_F(QMCRTracerTestFixture, rayTracing) {
    QMCRTracer.rayTracing(solarScene, 0);

    float *h_array = nullptr;
    Receiver *receiver = solarScene->getReceivers()[0];
    float *d_array = receiver->getDeviceImage();
    int2 resolution = receiver->getResolution();
    cout << "\nresolution: " << resolution.x << ", " << resolution.y << endl;
    global_func::gpu2cpu(h_array, d_array, resolution.x * resolution.y);

    float sum = 0.0f;
    for (int r = 0; r < resolution.y; ++r) {
        std::cout << std::endl;
        for (int c = 0; c < resolution.x; ++c) {
            sum += h_array[r * resolution.x + c];
            std::cout << h_array[r*resolution.x + c] << ' ';
        }
    }
    std::cout << "\nSum: " << sum << std::endl;
}