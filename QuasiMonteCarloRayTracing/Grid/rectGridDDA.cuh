//
// Created by feng on 19-4-14.
//

#ifndef SOLARENERGYRAYTRACING_RECTGRIDDDA_CUH
#define SOLARENERGYRAYTRACING_RECTGRIDDDA_CUH

#include "../RayTracingArgumentStruct.h"
#include "global_constant.h"
#include "cuda_runtime.h"
#include "RectangleGrid.cuh"


namespace rectGridDDA{
    template<typename T>
    inline __host__ __device__ T absDivide(const T &denominator, const T &numerator){   //分母，分子
        if(numerator <= Epsilon && numerator >= -Epsilon)
            return T(INT_MAX);
        return abs(denominator / numerator);
    }

    inline __host__ __device__ float  calTMax(float dir, float interval, int current_index, float current_pos){
        return abs(float(current_index + (dir >= 0)) * interval - current_pos);
    }

    inline __host__ __device__ bool equal(float n1, float n2){
        return (n1 < n2 + Epsilon) && (n1 > n2 - Epsilon);
    }

    inline __host__ __device__ bool less(float n1, float n2){
        return n1 < n2 - Epsilon;
    }

    /**
     * Intersect with heliostats in this rectangle grid.
     */
    __host__ __device__ bool intersect(const float3 &origin, const float3 &dir, const float3 *d_heliostat_vertexes,
                                        const int *d_grid_heliostat_match, int start_id, int end_id,
                                        int heliostat_id, int numberOfSubHeliostat);

    /**
     * 3D-DDA
     */
    __host__ __device__ bool collision(const float3 &origin, const float3 &dir, const RectGrid &rectGrid,
                                       const float3 *d_subheliostat_vertexes,
                                       const HeliostatArgument &heliostatArgument);

}


#endif //SOLARENERGYRAYTRACING_RECTFRIDDDA_CUH