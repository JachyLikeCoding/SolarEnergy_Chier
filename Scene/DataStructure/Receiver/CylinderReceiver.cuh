//
// Created by feng on 19-4-20.
//

#ifndef SOLARENERGY_CHIER_CYLINDERRECEIVER_CUH
#define SOLARENERGY_CHIER_CYLINDERRECEIVER_CUH

#include <math.h>
#include "Receiver.cuh"
#include "vector_arithmetic.cuh"
#include "global_constant.h"
#include <stdio.h>

/**
 * PS:
 *  size_:
 *      size_.x is the radius of cylinder
 *      size_.y is the height of cylinder
 *      size_.z has no meaning
 */

class CylinderReceiver : public Receiver{
public:
    __device__ __host__ CylinderReceiver(){}

    __device__ __host__ bool GIntersect(const float3 &orig, const float3 &dir, float &t, float &u, float &v){
        //If the origin is inside the cylinder, it won't intersect with it
        if(isInsideCylinder(orig))
            return false;

        float2 Ro = make_float2(pos_.x - orig.x, pos_.z - orig.z);
        float tp = dot(Ro, normalize(make_float2(dir.x, dir.z)));
        float delta = dot(Ro, Ro) - tp * tp;

        // Return false if:
        //  1) The direction is different
        //  2) Has no intersection
        float R2 = size_.x * size_.x;   // radius ^ 2
        if(tp < -Epsilon || delta > R2){
            return false;
        }

        float t_plus = delta <= 0.0f ? size_.x : sqrt(R2 - delta);
        t = (tp - t_plus) / length(make_float2(dir.x, dir,z));

        float3 intersect_pos = t * dir + orig;
        u = (intersect_pos.y - pos_.y) / size_.y + 0.5f;
        if(u < 0.0f || u > 1.0f){
            return false;
        }

        float2 intersect_origin_dir = make_float2(intersect_pos.x - pos_.x, intersect_pos.z - pos_.z);
        intersect_origin_dir = normalize(intersect_origin_dir); // (cosine, sine)

        if(intersect_origin_dir.x < -1 || intersect_origin_dir.x > 1){
            printf("\nError occurs on intersect position: %f, %f, %f\n", intersect_pos.x, intersect_pos.y, intersect_pos.z);
        }

        v = acosf(intersect_origin_dir.x) / (2 * M_PI);

        if(intersect_origin_dir.y < 0){
            v = 1 - v;
        }
        return true;
    }


    virtual void CInit(int pixel_per_meter_for_receiver);
    virtual void Cset_resolution(int pixel_per_meter_for_receiver);
    virtual float3 getFocusCenter(const float3 &heliostat_position);


private:
    __device__ __host__ bool isInsideCylinder(const float3 &orig){
        float2 l = make_float2(orig.x - pos_.x, orig.z - pos_.z);
        return dot(l, l) <= size_.x * size_.x;
    }

};



#endif //SOLARENERGY_CHIER_CYLINDERRECEIVER_CUH
