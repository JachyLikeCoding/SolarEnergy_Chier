//
// Created by feng on 19-3-28.
//

#ifndef SOLARENERGY_CHIER_RAYTRACINGPIPELINE_H
#define SOLARENERGY_CHIER_RAYTRACINGPIPELINE_H

#include <string>
#include "DataStructure/Receiver/Receiver.cuh"

class RayTracingPipeline{
public:
    static void rayTracing(int argc, char *argv[]);

private:
    static void saveReceiverResult(Receiver *receiver, std::string pathAndName);
};


#endif //SOLARENERGY_CHIER_RAYTRACINGPIPELINE_H
