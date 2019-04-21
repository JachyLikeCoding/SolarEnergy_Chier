//
// Created by feng on 19-4-21.
//

#include "receiverIntersectUtil.h"


void changeSunLightAndPerturbationToParallel(Sunray *sunray){
    if(sunray == nullptr)
        return;

    // Generate parallel lights and transfer them ro device
    int size = sunray->getNumOfSunshapeGroups() * sunray->getNumOfSunshapeLightsPerGroup();
    float3 *h_samplelights_and_perturbations = new float3[size];

    for(int i = 0; i < size; ++i){
        h_samplelights_and_perturbations[i] = make_float3(0.0f, 1.0f, 0.0f);
    }

    float3 *d_samplelights = nullptr;
    float3 *d_perturbations = nullptr;
    global_func::cpu2gpu(d_samplelights, h_samplelights_and_perturbations, size);
    global_func::cpu2gpu(d_perturbations, h_samplelights_and_perturbations, size);

    // Set the perturbation and sample lights of sunray
    sunray->CClear();
    sunray->setDevicePerturbation(d_perturbations);
    sunray->setDeviceSampleLights(d_samplelights);

    // Clean up
    delete[] h_samplelights_and_perturbations;
    h_samplelights_and_perturbations = nullptr;
}


std::vector<float> deviceArray2Vector(float *d_array, int size){
    vector<float> ans(size, 0.0f);
    float *h_array = nullptr;

    global_func::gpu2cpu(h_array, d_array, size);

    for(int i = 0; i < size; ++i){
        ans[i] = h_array[i];
    }

    delete[] h_array;
    h_array = nullptr;
    return ans;
}