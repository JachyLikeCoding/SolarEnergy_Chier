//
// Created by feng on 19-4-15.
//

#ifndef SOLARENERGY_CHIER_RECEIVERINTERSECTIONUTIL_CUH
#define SOLARENERGY_CHIER_RECEIVERINTERSECTIONUTIL_CUH

#include "vector_arithmetic.cuh"

/**
 * ηaα  =  | 0.99331 − 1.176 · 10^ -4 ⋅ d + 1.97 ⋅ 10^ −8 ⋅ d^2 , d <= 1000
            | e^ -0.0001106 ⋅ d , d > 1000
 * @param d the distance between origin and receiver. /m
 * @return ηaα
 */

inline __host__ __device__ float eta_aAlpha(const float &d){
    if(d <= 1000.0f)
        return 0.99331f - 0.0001176f * d + 1.97f * (1e-8f) * d * d;
    return expf(-0.0001106f * d);
}
/**
 * Eray = (ID ⋅ cosφ ⋅ ρ ⋅ Shsub / Nc ⋅ Srsub ) ⋅ ηaα
 * @param distance
 * @param dir
 * @param normal
 * @param factor
 * @return
 */

inline __host__ __device__ float calEnergy(float distance, float3 dir, float3 normal, float factor){
    //      cos(dir, normal)       *  ηaα                 *  (DNI * Ssub * reflective_rate / numberOfLightsPerGroup)
    return fabsf(dot(dir, normal)) * eta_aAlpha(distance) * factor;
}

#endif  //SOLARENERGY_CHIER_RECEIVERINTERSECTIONUTIL_CUH