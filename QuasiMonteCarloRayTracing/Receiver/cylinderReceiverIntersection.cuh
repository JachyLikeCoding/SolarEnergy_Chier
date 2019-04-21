//
// Created by feng on 19-4-21.
//

#ifndef SOLARENERGY_CHIER_CYLINDERRECEIVERINTERSECTION_CUH
#define SOLARENERGY_CHIER_CYLINDERRECEIVERINTERSECTION_CUH

#include "CylinderReceiver.cuh"

namespace cylinderReceiverIntersect{
    __device__ void receiver_drawing(CylinderReceiver &cylinderReceiver,
            const float3 &orig, const float3 &dir, const float3 &normal, float factor);
}

#endif //SOLARENERGY_CHIER_CYLINDERRECEIVERINTERSECTION_CUH