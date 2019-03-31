//
// Created by feng on 19-3-31.
// PS: Define the rectangle receiver data structure.
//

#ifndef SOLARENERGY_CHIER_RECTANGLE_RECEIVER_CUH
#define SOLARENERGY_CHIER_RECTANGLE_RECEIVER_CUH


#include "Receiver.cuh"
#include "global_function.cuh"

class RectangleReceiver : public Receiver{
public:
    __device__ __host__ RectangleReceiver(){}

    RectangleReceiver(const RectangleReceiver &rectangle_receiver) : Receiver(rectangle_receiver);

    //TODO: ADD TESTS
    __device__ __host__ bool GIntersect(const float3 &origin, const float3 &dir, float &t, float &u, float &v){
        return global_func::rayParallelogramIntersect(origin, dir, rect_vertexes_[0], rect_vertexes_[1], rect_vertexes_[3], t, u, v);
    }

    virtual void CInit(int pixel_per_meter_for_receiver);

    virtual float3 getFocusCenter(const float3 &heliostat_position);

    void setFocusCenter();

    float3 getRectangleVertex(int index) const;
    float3 getLocalNormal() const;


private:
    float3 rect_vertexes_[4];       //Four vertexes of rectangle
    float3 local_normal_;
    float3 focus_center_;

    void Cinit_vertex();
    void Cset_local_normal();           //Set local normal
    void Cset_local_vertex();           //Set local vertex
    void Cset_world_vertex();           //Set world vertex
    virtual void Cset_resolution(int pixel_per_meter_for_receiver);

};



















#endif //SOLARENERGY_CHIER_RECTANGLE_RECEIVER_CUH