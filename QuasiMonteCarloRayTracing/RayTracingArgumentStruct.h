//
// Created by feng on 19-4-14.
//

#ifndef SOLARENERGY_CHIER_RAYTRACINGARGUMENTSTRUCT_H
#define SOLARENERGY_CHIER_RAYTRACINGARGUMENTSTRUCT_H

#include "SolarScene.h"
#include "cuda_runtime.h"

struct SunrayArgument{
    float3 *d_samplelights;
    float3 *d_perturbations;
    int pool_size;
    int numberOfLightsPerGroup;
    float3 sunray_direction;

    SunrayArgument() : d_samplelights(nullptr), d_perturbations(nullptr){}

    SunrayArgument(float3 *samplelights, float3 *perturbations, int pool_size_, int lightsPerGroup, float3 sunray_dir) :
            d_samplelights(samplelights), d_perturbations(perturbations), pool_size(pool_size_),
            numberOfLightsPerGroup(lightsPerGroup), sunray_direction(sunray_dir){}

    ~SunrayArgument(){
        d_perturbations = nullptr;
        d_samplelights = nullptr;
    }
};


struct HeliostatArgument{
    float3 *d_microHelio_origins;
    float3 *d_microHelio_normals;
    int *d_microHelio_groups;
    int numberOfMicroHeliostats;
    int subHeliostat_id;
    int numberOfSubHeliostats;

    HeliostatArgument() : d_microHelio_origins(nullptr), d_microHelio_normals(nullptr), d_microHelio_groups(nullptr),
                            numberOfMicroHeliostats(0), subHeliostat_id(0), numberOfSubHeliostats(0){}

    HeliostatArgument(float3 *microHelio_origins, float3 *microHelio_normals, int * microHelioGroups, int numberOfMicroHelio, int subHelio_Id, int numberOfSubHelio) :
            d_microHelio_origins(microHelio_origins), d_microHelio_normals(microHelio_normals), d_microHelio_groups(microHelioGroups),
            numberOfMicroHeliostats(numberOfMicroHelio), subHeliostat_id(subHelio_Id), numberOfSubHeliostats(numberOfSubHelio){}

    ~HeliostatArgument(){
        d_microHelio_origins = nullptr;
        d_microHelio_normals = nullptr;
        d_microHelio_groups = nullptr;
    }

    void CClear(){
        cudaFree(d_microHelio_origins);
        cudaFree(d_microHelio_normals);
        cudaFree(d_microHelio_groups);

        d_microHelio_origins = nullptr;
        d_microHelio_normals = nullptr;
        d_microHelio_groups = nullptr;
    }
};


#endif //SOLARENERGY_CHIER_RAYTRACINGARGUMENTSTRUCT_H
