//
// Created by feng on 19-4-16.
//
#ifndef SOLARENERGY_CHIER_RECTANGLERECEIVERRECTANGLEGRIDRAYTRACING_CUH
#define SOLARENERGY_CHIER_RECTANGLERECEIVERRECTANGLEGRIDRAYTRACING_CUH

#include "global_function.cuh"
#include "cuda_runtime.h"
#include "RayTracingArgumentStruct.h"
#include "RectangleReceiver.cuh"
#include "RectangleGrid.cuh"




void RectangleReceiverRectangleGridRayTracing(SunrayArgument &sunrayArgument, RectangleReceiver *rectangleReceiver,
                                         RectangleGrid *rectGrid, HeliostatArgument &heliostatArgument,
                                         float3 *d_subHeliostat_vertexes, float factor);


#endif //SOLARENERGY_CHIER_RECTANGLERECEIVERRECTANGLEGRIDRAYTRACING_CUH

