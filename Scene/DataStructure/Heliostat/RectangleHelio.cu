//
// Created by feng on 19-3-30.
// PS: Define the rectangle heliostat data structure.
//
#include <iostream>
#include "RectangleHelio.cuh"
#include "global_function.cuh"
using namespace std;

std::vector<float> RectangleHelio::getSurfaceProperty(){
    return std::vector<float>(6, -1.0f);
}

void RectangleHelio::CGetSubHeliostatVertexes(std::vector<float3> &subHeliostatVertexes){
    subHeliostatVertexes.push_back(vertex_[0]);
    subHeliostatVertexes.push_back(vertex_[1]);
    subHeliostatVertexes.push_back(vertex_[2]);
}

void RectangleHelio::setSize(float3 size) {
    size_ = size;
}


void host_map_microhelio_center_and_normal(int myId, float3 *d_microhelio_centers, float3 *d_microhelio_normals,
                                          float3 normal, float3 helio_size, const int2 row_col, const int2 sub_row_col,
                                          const float pixel_length, const float2 gap, const float3 world_pos, const size_t map_size){
    if(myId >= map_size)
        return;

    int row = myId / (row_col.y * sub_row_col.y);
    int col = myId % (row_col.y * sub_row_col.y);

    int block_row = row / sub_row_col.x;
    int block_col = col / sub_row_col.y;

    //1. Generate local micro-heliostats' centers
    d_microhelio_centers[myId].x = col * pixel_length + block_col * gap.x + pixel_length / 2 - helio_size.x / 2;
    d_microhelio_centers[myId].y = helio_size.y / 2;
    d_microhelio_centers[myId].z = row * pixel_length + block_row * gap.y + pixel_length / 2 - helio_size.z / 2;

    //2. Generate micro-heliostats' normals
    d_microhelio_normals[myId] = normal;

    //3. Transform local micro-heliostat center to world position.
    float3 local = d_microhelio_centers[myId];
    local = global_func::local2world_rotate(local, normal);
    local = global_func::translate(local, world_pos);
    d_microhelio_centers[myId] = local;
}


namespace rectangle_heliostat{
    __global__ void map_microhelio_center_and_normal(float3 *d_microhelio_centers, float3 *d_microhelio_normals,
                                                    float3 normal, float3 helio_size, const int2 row_col, const int2 sub_row_col,
                                                    const float pixel_length, const float2 gap, const float3 world_pos, const size_t map_size){
        int myId = global_func::getThreadID();
        if(myId >= map_size)
            return;

        int row = myId / (row_col.y * sub_row_col.y);
        int col = myId % (row_col.y * sub_row_col.y);

        int block_row = row / sub_row_col.x;
        int block_col = col / sub_row_col.y;

        //1. Generate local micro-heliostats' centers
        d_microhelio_centers[myId].x = col * pixel_length + block_col * gap.x + pixel_length / 2 - helio_size.x / 2;
        d_microhelio_centers[myId].y = helio_size.y / 2;
        d_microhelio_centers[myId].z = row * pixel_length + block_row * gap.y + pixel_length / 2 - helio_size.z / 2;

        //2. Generate micro-heliostats' normals
        d_microhelio_normals[myId] = normal;

        //3. Transform local micro-heliostat center to world position.
        float3 local = d_microhelio_centers[myId];
        local = global_func::local2world_rotate(local, normal);
        local = global_func::translate(local, world_pos);
        d_microhelio_centers[myId] = local;
    }
}


int RectangleHelio::CGetDiscreteMicroHelioOriginsAndNormals(float3 *&d_microhelio_centers,
                                                            float3 *&d_microhelio_normals) {
    float2 subhelio_row_col_length;
    subhelio_row_col_length.x = (size_.z - gap_.y * (row_col_.x - 1)) / float(row_col_.x);
    subhelio_row_col_length.y = (size_.x - gap_.x * (row_col_.y - 1)) / float(row_col_.y);
    //cout << "subhelio_row_col_length----------" << subhelio_row_col_length.x << ", " << subhelio_row_col_length.y << endl;

    int2 sub_row_col;
    sub_row_col.x = int(subhelio_row_col_length.x / pixel_length_);
    sub_row_col.y = int(subhelio_row_col_length.y / pixel_length_);
    //cout << "sub_row_col------" << sub_row_col.x << ", " << sub_row_col.y << endl;

    int map_size = row_col_.x * row_col_.y * sub_row_col.x * sub_row_col.y;
    //cout << "map_size-----" << map_size << endl;

    int nThreads;
    dim3 nBlocks;
    global_func::setThreadBlocks(nBlocks, nThreads, map_size);

    //Map micro-heliostat center  and normal in world position.
    if(d_microhelio_centers == nullptr){
        checkCudaErrors(cudaMalloc((void **) &d_microhelio_centers, sizeof(float3) * map_size));
    }
    if(d_microhelio_normals == nullptr){
        checkCudaErrors(cudaMalloc((void **) &d_microhelio_normals, sizeof(float3) * map_size));
    }

    rectangle_heliostat::map_microhelio_center_and_normal << < nBlocks, nThreads >> >
            (d_microhelio_centers, d_microhelio_normals, normal_, size_, row_col_, sub_row_col, pixel_length_, gap_, pos_, map_size);
    printf("map_size: ");
    printf("%d\t", map_size);
    return map_size;
}

