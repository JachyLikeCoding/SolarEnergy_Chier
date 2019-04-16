//
// Created by feng on 19-4-15.
//

#ifndef SOLARENERGY_CHIER_QUASIMONTECARLORAYTRACER_H
#define SOLARENERGY_CHIER_QUASIMONTECARLORAYTRACER_H

#include "SolarScene.h"
#include "RayTracingArgumentStruct.h"

class QuasiMonteCarloRayTracer{
public:
    void rayTracing(SolarScene *solarScene, int heliostat_id);

    /**
     *  Public following function just for test. Do not use them standalone.
     */
    void checkValidHeliostatIndex(SolarScene *solarScene, int heliostat_id);

    int receiverGridCombination(int receiver_type, int grid_type);

    HeliostatArgument generateHeliostatArgument(SolarScene *solarScene, int heliostat_id);

    SunrayArgument generateSunrayArgument(Sunray *sunray);

    int setFlatRectangleHeliostatVertexs(float3 *&d_heliostat_vertexs, std::vector<Heliostat *> &heliostats, int start_id, int end_id);

};

#endif //SOLARENERGY_CHIER_QUASIMONTECARLORAYTRACER_H
