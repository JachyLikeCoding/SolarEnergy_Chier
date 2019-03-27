//
// Created by feng on 19-3-27.
// PS: Define some functions that can be used.
//
#pragma once

#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "global_constant.h"
#include "check_cuda.h"
#include "vector_arithmetic.cuh"

namespace global_func{
    // Rotate: The first step(1 of 2) to transform the local coordinate to world coordinate.
    __host__ __device__
    inline float3 local2world_rotate(const float3 &d_local, const float3 &aligned_normal){
        if(fabs(aligned_normal.x) < Epsilon && fabs(aligned_normal.z) < Epsilon){
            return d_local;
        }

        float3 u, n, v;
        n = aligned_normal;
        u = cross(make_float3(0.0f, 1.0f, 0.0f), n);
        u = normalize(u);
        v = cross(u,n);
        v = normalize(v);

        float3 d_world = make_float3(d_local.x * u.x + d_local.y * n.x + d_local.z * v.x,
                                     d_local.x * u.y + d_local.y * n.y + d_local.z * v.y,
                                     d_local.x * u.z + d_local.y * n.z + d_local.z * v.z);
        return d_world;
    }


    //Translate: The second step(2 of 2) to transform the local coordinate to world coordinate.
    template <typename T>
    __host__ __device__
    inline T translate(const T &d_in, const T translate_vector){
        return d_in + translate_vector;
    }


    // Rotate matrix:
    // |cos		0		sin| |origin.x|     |cos * origin.x + sin * origin.z |
    // |0		1		0  | |origin.y|  =  |           origin.y             |
    // |-sin	0		cos| |origin.z|     |-sin * origin.x + cos * origin.z|
    __host__ __device__
    inline float3 rotateY(const float3 &origin, const float3 &old_dir, const float3 &new_dir){
        int dir = (cross(old_dir, new_dir).y > 0) ? 1:-1;   //when parallel to (0,1,0), sin = 0, no effect
        float cos = dot(old_dir, new_dir);
        float sin = dir * sqrtf(1 - cos * cos);
        return make_float3(cos * origin.x + sin * origin.z,
                            origin.y,
                            -sin * origin.x + cos * origin.z);
    }


    //Judge whether the ray intersects with the parallelogram (ractangle)
    /* Reference: Fast, Minimum Storage Ray Triangle Intersection
     * ray equation: orig + dir * t  (origin: orig, direction: dir)
     * A/B/C are three vertexes of the parallelogram and A is the angle subtend diagonal.
     * In rectangle, A = PI/2
     * orig + dir * t = (1-u-v)A + uB + vC
     *                   |t|
     * ==> |-D  B-A C-A| |u| = orig - A
     *                   |v|
     * Now use Cramer's Rule and the mixed product of vectors to solve the system :
     * Make E1 = B - A, E2 = C - A, T = orig - A
     * According to Cramer's Rule:
     *
     *       t =   _____1_____  |T E1 E2|
     *              |-D E1 E2|
     *
     *       U =   _____1_____  |-D T E2|
     *              |-D E1 E2|
     *
     *       V =   _____1_____  |-D E1 T|
     *              |-D E1 E2|
     *
     *  Combine these three solutions together:
     *      |t|                 | T E1 E2|
     *      |u| =  _____1_____  |-D  T E2|
     *      |v|     |-D E1 E2|  |-D E1  T|
     *  According to the mixed product of vectors: |a b c| = a x b · c = -a x c · b
     *
     *  it can be changed in this way:
     *      |t|                     |T x E1 · E2|
     *      |u| =   ______1______   |D x E2 · T |
     *      |v|     |D x E1 · E2|   |T x E1 · D |
     *  Make P = D x E2
     *       Q = T x E1
     *  Get the final formula:
     *      |t|                |Q · E2|
     *      |u| =   ___1____   |P · T |
     *      |v|     |P · E1|   |Q · D |
     */
    __host__ __device__
    inline bool rayParallelogramIntersect(const float3 &orig, const float3 &dir,
                                            const float3 &A, const float3 &B, const float3 &C,
                                            float &t, float &u, float &v){
        float3 AB = B - A;
        float3 AC = C - A;
        //vector p
        float3 p = cross(dir, AC);
        //determinant
        float det = dot(AB, p);

        //if determinant is close to 0, ray lies in plane of parallelogram
        if(fabsf(det) < Epsilon)
            return false;

        float3 T = orig - A;

        //Calculate u and make sure u <= 1
        u = dot(T, p) / det;
        if(u < 0 || u > 1)
            return false;

        //vector q
        float3 q = cross(T, AB);

        //Calculate v and make sure v <= 1
        v = dot(dir, q) / det;
        if(v < 0 || v > 1)
            return false;

        t = dot(AC, q) / det;
        if(t < Epsilon)
            return false;

        return true;
    }

    //copy from cpu to gpu
    template <typename T>
    inline void cpu2gpu(T *&d_out, T *&h_in, const size_t &size) {
        if(d_out == nullptr){
            checkCudaErrors(cudaMalloc((void **) &d_out, sizeof(T) * size));
        }
        checkCudaErrors(cudaMemcpy(d_out, h_in, sizeof(T) * size, cudaMemcpyHostToDevice));
    }

    //copy from gpu to cpu
    template <typename T>
    inline void gpu2cpu(T *&h_out, T *&d_in, const size_t &size){
        if(h_out == nullptr){
            h_out = new T[size];
        }
        checkCudaErrors(cudaMemcpy(h_out, d_in, sizeof(T) * size, cudaMemcpyDeviceToHost));
    }


    //After fixing the number of threads
    __host__ __device__
    inline bool setThreadBlocks(dim3 &nBlocks, int nThreads, size_t size, bool threadFixed){
        if(size > MAX_ALL_THREADS){
            printf("There are too many threads to cope with. Please use less threads.\n");
            return false;
        }

        int block_lastDim = (size + nThreads - 1) / nThreads;
        if(block_lastDim < MAX_BLOCK_SINGLE_DIM){
            nBlocks.x = block_lastDim;
            nBlocks.y = 1;
            nBlocks.z = 1;
            return true;
        }

        block_lastDim = (block_lastDim + MAX_BLOCK_SINGLE_DIM - 1) /  MAX_BLOCK_SINGLE_DIM;
        if(block_lastDim < MAX_BLOCK_SINGLE_DIM){
            nBlocks.x = MAX_BLOCK_SINGLE_DIM;
            nBlocks.y = block_lastDim;
            nBlocks.z = 1;
            return true;
        }else{
            nBlocks.x = MAX_BLOCK_SINGLE_DIM;
            nBlocks.y = MAX_BLOCK_SINGLE_DIM;
            nBlocks.z = (block_lastDim + MAX_BLOCK_SINGLE_DIM - 1) / MAX_BLOCK_SINGLE_DIM;
            return true;
        }
    }

    //fix the number of threads
    __host__ __device__
    inline bool setThreadBlocks(dim3 &nBlocks, int nThreads, const size_t size){
        nThreads = (MAX_THREADS <= size) ? MAX_THREADS : size;
        return setThreadBlocks(nBlocks, nThreads, size, true);
    }


    __host__ __device__
    inline unsigned long long int getThreadID(){
        //unique block index inside a 3D block grid
        const unsigned long long int blockId = blockIdx.x                       //1D
                                                + blockIdx.y * gridDim.x            //2D
                                                + gridDim.x * gridDim.y * blockIdx.z;   //3D
        //global unique thread index, block dimension uses only x-coordinate
        const unsigned long long int threadId = blockId * blockDim.x + threadIdx.x;
        return threadId;
    }

    __host__ __device__
    inline float3 angle2xyz(float2 d_angles){
        return make_float3(sinf(d_angles.x) * cosf(d_angles.y),
                            cosf(d_angles.x),
                            sinf(d_angles.x) * sinf(d_angles.y));
    }

    //Unroll and roll the index and address
    __host__ __device__
    inline int unroll_index(int3 index, int3 matrix_size){
        return index.x * matrix_size.y * matrix_size.z + index.y * matrix_size.z + index.z;
    }

    __host__ __device__
    inline int unroll_index(int2 index, int2 matrix_size){
        return index.x * matrix_size.y + index.y;
    }
}
