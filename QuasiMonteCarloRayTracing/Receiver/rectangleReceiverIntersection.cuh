//
// Created by feng on 19-4-15.
//

#ifndef SOLARENERGY_CHIER_RANCTANGLERECEIVERINTERSECTION_CUH
#define SOLARENERGY_CHIER_RANCTANGLERECEIVERINTERSECTION_CUH

#include "vector_arithmetic.cuh"
#include "receiverIntersectionUtil.cuh"
#include "RectangleReceiver.cuh"


namespace rectangleReceiverIntersect{
    __device__ void receiver_drawing(RectangleReceiver &rectangleReceiver,
                                        const float3 &origin,
                                        const float3 &dir,
                                        const float3 &normal,
                                        float factor);
}

#endif  //SOLARENERGY_CHIER_RANCTANGLERECEIVERINTERSECTION_CUH
