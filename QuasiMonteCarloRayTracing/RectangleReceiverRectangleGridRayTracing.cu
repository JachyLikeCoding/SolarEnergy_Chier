#include "RectangleReceiverRectangleGridRayTracing.cuh"
#include "Receiver/rectangleReceiverIntersection.cuh"
#include "Grid/rectGridDDA.cuh"


__global__ void map_raytracing(SunrayArgument sunrayArgument, RectangleReceiver rectangleReceiver,
                                RectangleGrid rectangleGrid, HeliostatArgument heliostatArgument,
                               float3 *d_subheliostat_vertexes, float factor){
    long long myId = global_func::getThreadID();
    if(myId >= heliostatArgument.numberOfMicroHeliostats * sunrayArgument.numberOfLightsPerGroup){
        return;
    }

    // Step 1: whether the incident light is shadowed by other heliostats.  判断入射光
    int address = ( heliostatArgument.d_microHelio_groups[myId / sunrayArgument.numberOfLightsPerGroup] + myId % sunrayArgument.numberOfLightsPerGroup )
                        % sunrayArgument.pool_size;
    float3 dir = global_func::local2world_rotate(sunrayArgument.d_samplelights[address], -sunrayArgument.sunray_direction);
    float3 origin = heliostatArgument.d_microHelio_origins[myId / sunrayArgument.numberOfLightsPerGroup];

    if(rectGridDDA::collision(origin, dir, rectangleGrid, d_subheliostat_vertexes, heliostatArgument)){
        return;
    }

    // Step 2: whether the reflect light is shadowed by other heliostats. 判断反射光
    float3 normal = heliostatArgument.d_microHelio_normals[myId / sunrayArgument.numberOfLightsPerGroup];
    address = (heliostatArgument.d_microHelio_groups[(myId / sunrayArgument.numberOfLightsPerGroup + 1) % sunrayArgument.pool_size] + myId % sunrayArgument.numberOfLightsPerGroup)
                    % sunrayArgument.pool_size;
    normal = global_func::local2world_rotate(sunrayArgument.d_perturbations[address], normal);
    normal = normalize(normal);

    dir = normalize(reflect(-dir, origin));

    if(rectGridDDA::collision(origin, dir, rectangleGrid, d_subheliostat_vertexes, heliostatArgument)){
        return;
    }

    // Step 3: intersect with receiver. 和接收器求交，计算能量
    rectangleReceiverIntersect::receiver_drawing(rectangleReceiver, origin, dir, normal, factor);

}



void RectangleReceiverRectangleGridRayTracing( SunrayArgument &sunrayArgument,
                                                RectangleReceiver *rectangleReceiver,
                                                RectangleGrid *rectangleGrid,
                                                HeliostatArgument *heliostatArgument,
                                                float3 *d_subHeliostat_vertexes,
                                                float factor ){
    int nThreads = 512;
    dim3 nBlocks;
    global_func::setThreadBlocks(nBlocks, nThreads, heliostatArgument->numberOfMicroHeliostats * sunrayArgument.numberOfLightsPerGroup, true);

    map_raytracing << < nBlocks, nThreads >> >
                                 (sunrayArgument, *rectangleReceiver, *rectangleGrid, heliostatArgument, d_subHeliostat_vertexes, factor);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
