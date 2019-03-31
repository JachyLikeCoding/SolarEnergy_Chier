//
// Created by feng on 19-3-30.
//

#ifndef SOLARENERGY_CHIER_DATASTRUCTUREUTIL_H
#define SOLARENERGY_CHIER_DATASTRUCTUREUTIL_H

#include "vector_types.h"

inline bool Float3Eq(float3 n1, float3 n2, float gap){
    return (n1.x > n2.x - gap && n1.x < n1.x + gap)
       && (n1.y > n2.y - gap && n1.y < n2.y + gap)
       && (n1.z > n2.z - gap && n1.z < n2.z + gap);
}

#endif //SOLARENERGY_CHIER_DATASTRUCTUREUTIL_H
