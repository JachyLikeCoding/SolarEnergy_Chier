//
// Created by feng on 19-4-24.
//

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "FocusFlatRectangleHelio.cuh"
#include "global_function.cuh"
#include "DataStructureUtil.h"
#include "SceneLoader.h"

class FocusFlatRectangleHelioFixture : public ::testing::Test {
private:
    void setLocalCenterNormals() {
        localCenterNormals.push_back(make_float3(0.3f, 1.0f, 0.325f));
        localCenterNormals.push_back(make_float3(0.0f, 1.0f, 0.325f));
        localCenterNormals.push_back(make_float3(-0.3f, 1.0f, 0.325f));

        localCenterNormals.push_back(make_float3(0.3f, 1.0f, -0.325f));
        localCenterNormals.push_back(make_float3(0.0f, 1.0f, -0.325f));
        localCenterNormals.push_back(make_float3(-0.3f, 1.0f, -0.325f));

        for(int i=0;i<localCenterNormals.size();++i) {
            localCenterNormals[i] = normalize(localCenterNormals[i]);
        }
    }

    void setLocalCenterPositions() {
        localCenterPositions.push_back(make_float3(-0.6f, 0.245625f, -0.65f));
        localCenterPositions.push_back(make_float3(0.0f, 0.155625f, -0.65f));
        localCenterPositions.push_back(make_float3(0.6f, 0.245625f, -0.65f));

        localCenterPositions.push_back(make_float3(-0.6f, 0.245625f, 0.65f));
        localCenterPositions.push_back(make_float3(0.0f, 0.155625f, 0.65f));
        localCenterPositions.push_back(make_float3(0.6f, 0.245625f, 0.65f));
    }

protected:
    void SetUp() {
        heliostatLocation = make_float3(10, 2, 6);

        focusFlatRectangleHelio.setSize(make_float3(1.7f, 0.1f, 2.3f));
        focusFlatRectangleHelio.setGap(make_float2(0.1f, 0.3f));
        focusFlatRectangleHelio.setRowAndColumn(make_int2(2, 3));
        focusFlatRectangleHelio.setPixelLength(0.5f);
        focusFlatRectangleHelio.setNormal(make_float3(0.0f, 1.0f, 0.0f));
        focusFlatRectangleHelio.setPosition(heliostatLocation);

        std::vector<float> surface_property(6,-1.0f);
        surface_property[0] = 1;
        focusFlatRectangleHelio.setSurfaceProperty(surface_property);

        setLocalCenterNormals();
        setLocalCenterPositions();
    }

public:
    FocusFlatRectangleHelio focusFlatRectangleHelio;
    float3 heliostatLocation;
    std::vector<float3> localCenterNormals;
    std::vector<float3> localCenterPositions;

    std::vector<float3> GPUArrayConvert2vector(float3 *d_array, int size) {
        float3 *h_array = nullptr;
        global_func::gpu2cpu(h_array, d_array, size);

        std::vector<float3> ans;
        for (int i = 0; i < size; ++i) {
            ans.push_back(h_array[i]);
        }

        //clear
        delete[] h_array;
        h_array = nullptr;

        return ans;
    }

    float3 microHeliostatPointTransfer(float3 position, float3 microhelioLocalNormal, float3 microhelioLocalPosition) {
        position = global_func::local2world_rotate(position, microhelioLocalNormal);
        position = global_func::translate(position, microhelioLocalPosition);
        position = global_func::local2world_rotate(position, focusFlatRectangleHelio.getNormal());
        return global_func::translate(position, focusFlatRectangleHelio.getPosition());
    }

    float3 microHeliostatPointLocalTransfer(float3 position, float3 microhelioLocalNormal, float3 microhelioLocalPosition) {
        position = global_func::local2world_rotate(position, microhelioLocalNormal);
        position = global_func::translate(position, microhelioLocalPosition);
        return global_func::local2world_rotate(position, focusFlatRectangleHelio.getNormal());
    }

    float3 negitiveX(float3 n) {
        return make_float3(-n.x, n.y, n.z);
    }

    float3 negitiveZ(float3 n) {
        return make_float3(n.x, n.y, -n.z);
    }
};

MATCHER_P(FloatNearPointwise, gap, "Check whether two float3 objects are almost the same") {
    float3 n1 = std::get<0>(arg);
    float3 n2 = std::get<1>(arg);

    *result_listener << "\nExpect value: (" << n1.x << ", " << n1.y << ", " << n1.z << ")\n";
    *result_listener << "Got value: (" << n2.x << ", " << n2.y << ", " << n2.z << ")";
    return Float3Eq(n1, n2, (float) gap);
}

/**
 * Test for CSetNormalAndRotate
 *
 * */


/**
 * Test for GPU local centers and normals(test before CGetSubHeliostatVertexes)
 *
 * */
TEST_F(FocusFlatRectangleHelioFixture, GPULocalCentersAndNormals) {
    std::vector<float3> calculatedSubHeliostatVertexes;
    focusFlatRectangleHelio.CGetSubHeliostatVertexes(calculatedSubHeliostatVertexes);
    std::vector<float3> calculatedGPULocalCenters = focusFlatRectangleHelio.getGPULocalCenters();
    EXPECT_THAT(localCenterPositions,
                ::testing::Pointwise(FloatNearPointwise(1e-3), calculatedGPULocalCenters));

    std::vector<float3> calculatedGPULocalNormals = focusFlatRectangleHelio.getGPULocalNormals();
    EXPECT_THAT(localCenterNormals,
                ::testing::Pointwise(FloatNearPointwise(1e-3), calculatedGPULocalNormals));
}

/**
 * Test for CGetSubHeliostatVertexes(test after GPULocalCentersAndNormals)
 *
 * */
TEST_F(FocusFlatRectangleHelioFixture, CGetSubHeliostatVertexes) {
    float3 vertexLeftBottom = make_float3(-0.25f, 0.0f, 0.5f);
    float3 vertexLeftUpper = make_float3(-0.25f, 0.0f, -0.5f);
    float3 vertexRightUpper = make_float3(0.25f, 0.0f, -0.5f);
    float3 vertexRightBottom = make_float3(0.25f, 0.0f, 0.5f);

    float3 subheliostat00VertexLeftBottom =
            microHeliostatPointLocalTransfer(vertexLeftBottom, localCenterNormals[0], localCenterPositions[0]);
    cout<<"localCenterPositions[0]: "<<localCenterPositions[0].x<<", "<<localCenterPositions[0].y<<", "<<localCenterPositions[0].z<<endl;
    float3 subheliostat00VertexLeftUpper =
            microHeliostatPointLocalTransfer(vertexLeftUpper, localCenterNormals[0], localCenterPositions[0]);
    float3 subheliostat00VertexRightUpper =
            microHeliostatPointLocalTransfer(vertexRightUpper, localCenterNormals[0], localCenterPositions[0]);
    float3 subheliostat00VertexRightBottom =
            microHeliostatPointLocalTransfer(vertexRightBottom, localCenterNormals[0], localCenterPositions[0]);

    float3 subheliostat01VertexLeftBottom =
            microHeliostatPointLocalTransfer(vertexLeftBottom, localCenterNormals[1], localCenterPositions[1]);
    float3 subheliostat01VertexLeftUpper =
            microHeliostatPointLocalTransfer(vertexLeftUpper, localCenterNormals[1], localCenterPositions[1]);
    float3 subheliostat01VertexRightUpper =
            microHeliostatPointLocalTransfer(vertexRightUpper, localCenterNormals[1], localCenterPositions[1]);
    float3 subheliostat01VertexRightBottom =
            microHeliostatPointLocalTransfer(vertexRightBottom, localCenterNormals[1], localCenterPositions[1]);

    std::vector<float3> expectLocalVertexPositions;
    // First Row
    expectLocalVertexPositions.push_back(subheliostat00VertexLeftBottom + heliostatLocation);
    expectLocalVertexPositions.push_back(subheliostat00VertexLeftUpper + heliostatLocation);
    expectLocalVertexPositions.push_back(subheliostat00VertexRightUpper + heliostatLocation);

    expectLocalVertexPositions.push_back(subheliostat01VertexLeftBottom + heliostatLocation);
    expectLocalVertexPositions.push_back(subheliostat01VertexLeftUpper + heliostatLocation);
    expectLocalVertexPositions.push_back(subheliostat01VertexRightUpper + heliostatLocation);

    expectLocalVertexPositions.push_back(negitiveX(subheliostat00VertexRightBottom) + heliostatLocation);
    expectLocalVertexPositions.push_back(negitiveX(subheliostat00VertexRightUpper) + heliostatLocation);
    expectLocalVertexPositions.push_back(negitiveX(subheliostat00VertexLeftUpper) + heliostatLocation);

    // Second Row
    expectLocalVertexPositions.push_back(negitiveZ(subheliostat00VertexLeftUpper) + heliostatLocation);
    expectLocalVertexPositions.push_back(negitiveZ(subheliostat00VertexLeftBottom) + heliostatLocation);
    expectLocalVertexPositions.push_back(negitiveZ(subheliostat00VertexRightBottom) + heliostatLocation);

    expectLocalVertexPositions.push_back(negitiveZ(subheliostat01VertexLeftUpper) + heliostatLocation);
    expectLocalVertexPositions.push_back(negitiveZ(subheliostat01VertexLeftBottom) + heliostatLocation);
    expectLocalVertexPositions.push_back(negitiveZ(subheliostat01VertexRightBottom) + heliostatLocation);

    expectLocalVertexPositions.push_back(negitiveX(negitiveZ(subheliostat00VertexRightUpper)) + heliostatLocation);
    expectLocalVertexPositions.push_back(negitiveX(negitiveZ(subheliostat00VertexRightBottom)) + heliostatLocation);
    expectLocalVertexPositions.push_back(negitiveX(negitiveZ(subheliostat00VertexLeftBottom)) + heliostatLocation);

    for(int i = 0; i < expectLocalVertexPositions.size(); ++i){
        cout << "expectLocalVertexPositions[" << i << "] = " << expectLocalVertexPositions[i].x << ", " << expectLocalVertexPositions[i].y << ", " << expectLocalVertexPositions[i].z << endl;
    }

    std::vector<float3> calculatedLocalVertexPositions;
    focusFlatRectangleHelio.CGetSubHeliostatVertexes(calculatedLocalVertexPositions);
    for (int i = 0; i < calculatedLocalVertexPositions.size(); ++i) {
        cout << "calculatedLocalVertexPositions[" << i <<"] = " << calculatedLocalVertexPositions[i].x << " " << calculatedLocalVertexPositions[i].y << " " << calculatedLocalVertexPositions[i].z << endl;
    }
    for (int i = 0; i < expectLocalVertexPositions.size(); ++i) {
        cout << "expectLocalVertexPositions[" << i <<"] = " << expectLocalVertexPositions[i].x << " " << calculatedLocalVertexPositions[i].y << " " << calculatedLocalVertexPositions[i].z << endl;
    }
    EXPECT_THAT(expectLocalVertexPositions,
                ::testing::Pointwise(FloatNearPointwise(1e-3), calculatedLocalVertexPositions));
}

/**
 * Test for CGetDiscreteMicroHelioOriginsAndNormals
 *
 * */
TEST_F(FocusFlatRectangleHelioFixture, CGetDiscreteMicroHelioOriginsAndNormals) {
    // Calculate micro-helistat center positions
    float3 microheliostatPositionUpper = make_float3(0.0f, 0.0f, -0.25f);
    float3 microheliostatPositionBottom = make_float3(0.0f, 0.0f, 0.25f);

    int id = global_func::unroll_index(make_int2(0, 0), focusFlatRectangleHelio.getRowAndColumn());

    float3 microheliostatPosition00 = microHeliostatPointLocalTransfer(microheliostatPositionUpper,
                                                                       localCenterNormals[id], localCenterPositions[id]);
    float3 microheliostatPosition10 = microHeliostatPointLocalTransfer(microheliostatPositionBottom,
                                                                       localCenterNormals[id], localCenterPositions[id]);

    id = global_func::unroll_index(make_int2(0, 1), focusFlatRectangleHelio.getRowAndColumn());

    float3 microheliostatPosition01 = microHeliostatPointLocalTransfer(microheliostatPositionUpper,
                                                                       localCenterNormals[id], localCenterPositions[id]);
    float3 microheliostatPosition11 = microHeliostatPointLocalTransfer(microheliostatPositionBottom,
                                                                       localCenterNormals[id], localCenterPositions[id]);

    std::vector<float3> expectLocalMicroHeliostatPositions;
    expectLocalMicroHeliostatPositions.push_back(microheliostatPosition00 + heliostatLocation);
    expectLocalMicroHeliostatPositions.push_back(microheliostatPosition01 + heliostatLocation);
    expectLocalMicroHeliostatPositions.push_back(negitiveX(microheliostatPosition00) + heliostatLocation);

    expectLocalMicroHeliostatPositions.push_back(microheliostatPosition10 + heliostatLocation);
    expectLocalMicroHeliostatPositions.push_back(microheliostatPosition11 + heliostatLocation);
    expectLocalMicroHeliostatPositions.push_back(negitiveX(microheliostatPosition10) + heliostatLocation);

    expectLocalMicroHeliostatPositions.push_back(negitiveZ(microheliostatPosition10) + heliostatLocation);
    expectLocalMicroHeliostatPositions.push_back(negitiveZ(microheliostatPosition11) + heliostatLocation);
    expectLocalMicroHeliostatPositions.push_back(negitiveX(negitiveZ(microheliostatPosition10)) + heliostatLocation);

    expectLocalMicroHeliostatPositions.push_back(negitiveZ(microheliostatPosition00) + heliostatLocation);
    expectLocalMicroHeliostatPositions.push_back(negitiveZ(microheliostatPosition01) + heliostatLocation);
    expectLocalMicroHeliostatPositions.push_back(negitiveX(negitiveZ(microheliostatPosition00)) + heliostatLocation);

    for(int i = 0; i < expectLocalMicroHeliostatPositions.size(); ++i){
        cout << "expectLocalMicroHeliostatPositions[" << i << "] = " << expectLocalMicroHeliostatPositions[i].x << ", " << expectLocalMicroHeliostatPositions[i].y << ", " << expectLocalMicroHeliostatPositions[i].z << endl;
    }
    // Calculate micro-helistat normal directions
    std::vector<float3> expectLocalMicroHeliostatNormal({
                                                                localCenterNormals[0], localCenterNormals[1], localCenterNormals[2],
                                                                localCenterNormals[0], localCenterNormals[1], localCenterNormals[2],
                                                                localCenterNormals[3], localCenterNormals[4], localCenterNormals[5],
                                                                localCenterNormals[3], localCenterNormals[4], localCenterNormals[5]});
    for(int i = 0; i < expectLocalMicroHeliostatNormal.size(); ++i){
        cout << "expectedLocalMicroHeliostatNormal[" << i << "] = " << expectLocalMicroHeliostatNormal[i].x << ", " << expectLocalMicroHeliostatNormal[i].y << ", " << expectLocalMicroHeliostatNormal[i].z << endl;
    }


    float3 *d_microhelio_centers = nullptr;
    float3 *d_microhelio_normals = nullptr;
    /**
     * Make sure the d_local_normals and d_local_centers are calculated
     * */
    std::vector<float3> calculatedLocalVertexPositions;
    focusFlatRectangleHelio.CGetSubHeliostatVertexes(calculatedLocalVertexPositions);
    focusFlatRectangleHelio.CGetDiscreteMicroHelioOriginsAndNormals(d_microhelio_centers, d_microhelio_normals);

    int N = expectLocalMicroHeliostatNormal.size();
    std::vector<float3> microhelio_centers_vector = GPUArrayConvert2vector(d_microhelio_centers, N);
    std::vector<float3> microhelio_normals_vector = GPUArrayConvert2vector(d_microhelio_normals, N);
    for(int i = 0; i < expectLocalMicroHeliostatNormal.size(); ++i){
        cout << "microhelio_centers_vector[" << i << "] = " << microhelio_centers_vector[i].x << ", " << microhelio_centers_vector[i].y << ", " << microhelio_centers_vector[i].z << endl;
    }
    cout << endl;
    for(int i = 0; i < microhelio_normals_vector.size(); ++i){
        cout << "microhelio_normals_vector[" << i << "] = " << microhelio_normals_vector[i].x << ", " << microhelio_normals_vector[i].y << ", " << microhelio_normals_vector[i].z << endl;
    }

    EXPECT_THAT(expectLocalMicroHeliostatPositions,
                ::testing::Pointwise(FloatNearPointwise(1e-3), microhelio_centers_vector));
    EXPECT_THAT(expectLocalMicroHeliostatNormal,
                ::testing::Pointwise(FloatNearPointwise(1e-3), microhelio_normals_vector));

    //clear
    checkCudaErrors(cudaFree(d_microhelio_centers));
    checkCudaErrors(cudaFree(d_microhelio_normals));
    d_microhelio_centers = nullptr;
    d_microhelio_normals = nullptr;
}


TEST_F(FocusFlatRectangleHelioFixture, LoadFocusFlatRectangle) {
    SolarScene *solarScene = SolarScene::GetInstance();
    SceneLoader *sceneLoader = new SceneLoader();

    std::string scenePath = "test_file/test_scene_good_for_focusFlatRectangleHelio.scn";
    EXPECT_TRUE(sceneLoader->SceneFileRead(solarScene, scenePath));
}