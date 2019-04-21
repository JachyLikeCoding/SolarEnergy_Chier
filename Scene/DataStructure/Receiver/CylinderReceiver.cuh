//
// Created by feng on 19-4-20.
//

#ifndef SOLARENERGY_CHIER_CYLINDERRECEIVER_CUH
#define SOLARENERGY_CHIER_CYLINDERRECEIVER_CUH

#include <iostream>
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

    /**
     * 1) Define equation:
     *      -ray equation: p(t) = (orig - pos) + t * dir     (1)
     *      -cylinder equation: z^2 + m^2 * x^2 = r^2     (r = size.x,  m = 1)    (2)
     * 2) combine equation (1) and (2):
     *      calculate delta: delta = b^2 - 4ac
     *          -if delta >= 0, may have intersection
     *          -if delta < 0, no intersection
     * 3) calculate t and point of intersection
     *          -if t < 0, wrong direction, so remain the positive solution
     *          -if intersect_point.z is not belonging to the range [0, size.z], not effective intersection and return false.
     */
    __device__ __host__ bool GIntersect(const float3 &orig, const float3 &dir, float &t, float &u, float &v){
        //If the origin is inside the cylinder, it won't intersect with it
        if(isInsideCylinder(orig)){
            printf("The origin is inside the cylinder!!!\n");
            return false;
        }

        // simplified expression and got : Ro^2 - tp^2 <= R2, represent delta >= 0
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
        // remain and calculate the positive solution
        float t_plus = delta <= 0.0f ? size_.x : sqrt(R2 - delta);
        t = (tp - t_plus) / length(make_float2(dir.x, dir.z));

        // calculate intersection point
        float3 intersect_pos = t * dir + orig;
        std::cout << "intersect_pos: ( " << intersect_pos.x << ", " << intersect_pos.y << ", " << intersect_pos.z << ")\n";
        u = (intersect_pos.y - pos_.y) / size_.y + 0.5f;
        std::cout << "u = " << u << "\n";
        // intersect_pos.y should belong to the range [0, h], h = size_.y
        if(u < 0.0f || u > 1.0f){
            return false;
        }

        float2 intersect_origin_dir = make_float2(intersect_pos.x - pos_.x, intersect_pos.z - pos_.z);
        intersect_origin_dir = normalize(intersect_origin_dir); // (cosine, sine)

        /**
         * TODO:ADD ERROR TESTS
         */
        if(intersect_origin_dir.x < -1 || intersect_origin_dir.x > 1){
            printf("\nError occurs on intersect position: %f, %f, %f\n", intersect_pos.x, intersect_pos.y, intersect_pos.z);
        }

        v = acosf(intersect_origin_dir.x) / (2 * M_PI);
        std::cout << "v1 = " << v << "\n";
        if(intersect_origin_dir.y < 0){
            v = 1 - v;
        }
        std::cout << "v2 = " << v << "\n";
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
