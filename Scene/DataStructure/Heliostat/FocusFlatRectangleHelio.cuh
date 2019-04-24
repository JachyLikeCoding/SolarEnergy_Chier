//
// Created by feng on 19-4-23.
// PS: Define the FocusFlatHeliostat data structure.
//

#ifndef SOLARENERGY_CHIER_FOCUSFLATHELIO_CUH
#define SOLARENERGY_CHIER_FOCUSFLATHELIO_CUH


#include "Heliostat.cuh"
#include "check_cuda.h"
#include "global_function.cuh"

namespace focusFlatRectangle_heliostat{
    __host__ __device__
    inline float3 focusFlatRectangleHeliostatLocal2WorldRotate(const float3 &d_local, const float3 &aligned_normal){
        // u : X
        // n is normal : Y
        // v : Z
        // Sample(world coordinate) = Sample(local) * Transform matrix
        // Transform matrix:
        //  |u.x    u.y     u.z|
        //  |n.x    n.y     n.z|
        //  |v.x    v.y     v.z|

        if(fabs(aligned_normal.x) < Epsilon && fabs(aligned_normal.z) < Epsilon){
            return d_local;
        }

        float3 u, n, v; // could be shared
        n = aligned_normal;

        u = cross(make_float3(0.0f, 1.0f, 0.0f), n);
        u = normalize(u);
        v = cross(u, n);
        v = normalize(v);
        if(n.z < 0){
            u = -u;
            v = -v;
        }

        float3 d_world = make_float3(d_local.x * u.x + d_local.y * n.x + d_local.z * v.x,
                                     d_local.x * u.y + d_local.y * n.y + d_local.z * v.y,
                                     d_local.x * u.z + d_local.y * n.z + d_local.z * v.z);
        return d_world;
    }
}


class FocusFlatRectangleHelio : public Heliostat{
private:
    float focus_length_;
    int read_method;    // choose read method : (0) calculate by analytic formula (1) read from input file
    float3 *d_local_normals;
    float3 *d_local_centers;

public:
    FocusFlatRectangleHelio() : d_local_centers(nullptr), d_local_normals(nullptr){}
    ~FocusFlatRectangleHelio(){
        if(d_local_normals){
            checkCudaErrors(cudaFree(d_local_normals));
            d_local_normals = nullptr;
        }

        if(d_local_centers){
            checkCudaErrors(cudaFree(d_local_centers));
            d_local_centers = nullptr;
        }
    }

    virtual int getSubHelioSize(){
        return row_col_.x * row_col_.y;
    }

    virtual void setSize(float3 size);

    virtual void CSetNormalAndRotate(const float3 &focus_center, const float3 &sunray_dir);

    virtual int CGetDiscreteMicroHelioOriginsAndNormals(float3 *&d_microhelio_centers, float3 *&d_microhelio_normals);

    virtual void CGetSubHeliostatVertexes(std::vector<float3> &SubHeliostatVertexes);

    virtual void setSurfaceProperty(const std::vector<float> &surface_property);

    virtual std::vector<float> getSurfaceProperty();

    float getFocusLength() const;
    void setFocusLength(float focus_length);

    std::vector<float3> getGPULocalNormals();
    void setGPULocalNormals(float3 *h_local_normals);
    void setGPULocalNormals(std::vector<float3> local_normals);

    std::vector<float3> getGPULocalCenters();

    void setGPULocalCenters(float3 *h_local_centers);
    void setGPULocalCenters(std::vector<float3> local_centers);
};

#endif //SOLARENERGY_CHIER_FOCUSFLATHELIO_CUH
