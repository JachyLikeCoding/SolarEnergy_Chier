//
// Created by feng on 19-3-29.
//

#include<iostream>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "RectangleGrid.cuh"
#include "RectangleHelio.cuh"
#include "global_function.cuh"

class RectangleGridFixture : public ::testing::Test{
public:
    RectangleGrid rectGrid;
    std::vector<Heliostat *> heliostats;
    Heliostat *h1, *h2, *h3, *h4, *h5;
    RectangleGridFixture() : h1(nullptr), h2(nullptr), h3(nullptr), h4(nullptr), h5(nullptr){}

    vector<int> convert2vector(int *array, int size){
        vector<int> ans;
        for(int i = 0; i < size; ++i){
            ans.push_back(array[i]);
        }
        return ans;
    }

protected:
    void SetUp(){
        /**
         * The scene is similar in test_file/test_rectangle.scn
         */
         //Rectangle Grid
        rectGrid.setGridType(0);
        rectGrid.setPosition(make_float3(-15.0f, 0.0f, 20.0f));
        rectGrid.setSize(make_float3(30.0f, 10.0f, 20.0f));
        rectGrid.setInterval(make_float3(10.0f, 10.0f, 10.0f));
        rectGrid.setNumberOfHeliostats(5);
        rectGrid.setHeliostatType(0);
        rectGrid.setStartHeliostatIndex(0);
        rectGrid.Cinit();

        //Heliostats
        float3 helio_size = make_float3(4.0f, 0.1f, 3.0f);

        h1 = new RectangleHelio();
        h1->setSize(helio_size * 1.5f);
        h1->setPosition(make_float3(-8.0f, 5.0f, 25.0f));

        h2 = new RectangleHelio();
        h2->setSize(helio_size * 0.75f);
        h2->setPosition(make_float3(10.0f, 5.0f, 22.50f));

        h3 = new RectangleHelio();
        h3->setSize(helio_size * 0.75f);
        h3->setPosition(make_float3(10.0f, 5.0f, 27.50f));

        h4 = new RectangleHelio();
        h4->setSize(helio_size);
        h4->setPosition(make_float3(-10.0f, 5.0f, 35.0f));

        h5 = new RectangleHelio();
        h5->setSize(helio_size);
        h5->setPosition(make_float3(0.0f, 5.0f, 35.0f));

        heliostats.push_back(h1);
        heliostats.push_back(h2);
        heliostats.push_back(h3);
        heliostats.push_back(h4);
        heliostats.push_back(h5);
    }

    void TearDown(){
        delete(h1);
        delete(h2);
        delete(h3);
        delete(h4);
        delete(h5);
    }
};


TEST_F(RectangleGridFixture, CGridHelioMatchGoodExample1){
    //Test function
    rectGrid.CGridHelioMatch(heliostats);

    //Check the result as our expect
    int *h_grid_helio_match = nullptr;
    int *h_grid_helio_index = nullptr;

    int *d_grid_helio_match = rectGrid.getDeviceGridHeliostatMatch();
    int *d_grid_helio_index = rectGrid.getDeviceGridHeliostatIndex();
    cout << "rectGrid.getSubGridNumber(): " << rectGrid.getSubGridNumber().x <<", "<< rectGrid.getSubGridNumber().y <<", "<< rectGrid.getSubGridNumber().z << endl;

    int size = rectGrid.getSubGridNumber().x * rectGrid.getSubGridNumber().y * rectGrid.getSubGridNumber().z + 1;
    global_func::gpu2cpu(h_grid_helio_match, d_grid_helio_match, rectGrid.getNumberOfGridHeliostatMatch());
    global_func::gpu2cpu(h_grid_helio_index, d_grid_helio_index, size);

    std::cout << "======Index======" << std::endl;
    vector<int> index;
    index = convert2vector(h_grid_helio_index, size);
    for(int i = 0; i < index.size(); i++)
        cout << index[i] <<" ";
    cout << endl;
    EXPECT_THAT(convert2vector(h_grid_helio_index, size), testing::ElementsAreArray({0,1,2,3,4,6,6}));

    std::cout << "======Match====== " << std::endl;
    vector<int> match;
    match = convert2vector(h_grid_helio_match, rectGrid.getNumberOfGridHeliostatMatch());
    for(int i = 0; i < match.size(); i++)
        cout << match[i] <<" ";
    cout << endl;
    EXPECT_THAT(convert2vector(h_grid_helio_match, rectGrid.getNumberOfGridHeliostatMatch()), testing::ElementsAreArray({0,3,0,4,1,2}));
}


TEST_F(RectangleGridFixture, CGridHelioMatchGoodExample12){
    /**
     * The scene is similar in test_file/test_rectgrid.scn
     */
     h1->setNumberOfSubHelio(1);
     h2->setNumberOfSubHelio(2);
     h3->setNumberOfSubHelio(3);
     h4->setNumberOfSubHelio(4);
     h5->setNumberOfSubHelio(5);

     //Test function
     rectGrid.CGridHelioMatch(heliostats);

     //Check the result as our expect.
     int *h_grid_helio_match = nullptr;
     int *h_grid_helio_index = nullptr;

     int *d_grid_helio_match = rectGrid.getDeviceGridHeliostatMatch();
     int *d_grid_helio_index = rectGrid.getDeviceGridHeliostatIndex();
     int size = rectGrid.getSubGridNumber().x * rectGrid.getSubGridNumber().y * rectGrid.getSubGridNumber().z + 1;

     global_func::gpu2cpu(h_grid_helio_index, d_grid_helio_index, size);
     global_func::gpu2cpu(h_grid_helio_match, d_grid_helio_match, rectGrid.getNumberOfGridHeliostatMatch());

    std::cout << "======Index======" << std::endl;
    vector<int> index;
    index = convert2vector(h_grid_helio_index, size);
    for(int i = 0; i < index.size(); i++)
        cout << index[i] <<" ";
    cout << endl;
    EXPECT_THAT(convert2vector(h_grid_helio_index, size), testing::ElementsAreArray({0, 1, 5, 6, 11, 16, 16}));

    std::cout << "======Match====== " << std::endl;
    vector<int> match;
    match = convert2vector(h_grid_helio_match, rectGrid.getNumberOfGridHeliostatMatch());
    for(int i = 0; i < match.size(); i++)
        cout << match[i] <<" ";
    cout << endl;
    EXPECT_THAT(convert2vector(h_grid_helio_match, rectGrid.getNumberOfGridHeliostatMatch()),
            testing::ElementsAreArray({0, 6, 7, 8, 9, 0, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5}));
}

