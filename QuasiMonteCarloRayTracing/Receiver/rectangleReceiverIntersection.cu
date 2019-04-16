//
// Created by feng on 19-4-15.
//
#include "rectangleReceiverIntersection.cuh"

__device__ void rectangleReceiverIntersect::receiver_drawing(RectangleReceiver &rectangleReceiver, const float3 &origin,
                                                             const float3 &dir, const float3 &normal, float factor) {
    // Step 1: Intersect with receiver
    float t, u, v;
    if(!rectangleReceiver.GIntersect(origin, dir, t, u, v)){
        return;
    }

    // Step 2: Calculate the sun energy
    float energy = calEnergy(t, dir, normal, factor);

    // Step 3: Add the energy to the intersect position
    // Intersect position
    int2 row_col = make_int2(u * rectangleReceiver.getResolution().y,
                             v * rectangleReceiver.getResolution().x);
    int address = row_col.x * rectangleReceiver.getResolution().x + row_col.y; // col_row.y + col_row.x * resolution.y
    float *image = rectangleReceiver.getDeviceImage();
    atomicAdd(&(image[address]), energy);   //CUDA atomic原子操作 —— 在kernel程序中做统计累加， 需使用原子操作

}