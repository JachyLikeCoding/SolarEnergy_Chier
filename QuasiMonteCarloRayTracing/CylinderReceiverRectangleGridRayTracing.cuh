//
// Created by feng on 19-4-21.
//
#ifndef SOLARENERGY_CHIER_CYLINDERRECEIVERRECTANGLEGRIDRAYTRACING_CUH
#define SOLARENERGY_CHIER_CYLINDERRECEIVERRECTANGLEGRIDRAYTRACING_CUH

#include "global_function.cuh"
#include "cuda_runtime.h"
#include "RayTracingArgumentStruct.h"
#include "CylinderReceiver.cuh"
#include "RectangleGrid.cuh"




void CylinderReceiverRectangleGridRayTracing(SunrayArgument &sunrayArgument, CylinderReceiver *cylinderReceiver,
                                              RectangleGrid *rectGrid, HeliostatArgument &heliostatArgument,
                                              float3 *d_subHeliostat_vertexes, float factor);


#endif //SOLARENERGY_CHIER_CYLINDERRECEIVERRECTANGLEGRIDRAYTRACING_CUH

