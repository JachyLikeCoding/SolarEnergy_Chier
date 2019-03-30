//
// Created by feng on 19-3-30.
// PS: Define the rectangle heliostat data structure.
//
#ifndef SOLARENERGY_CHIER_RECTANGLEHELIO_CUH
#define SOLARENERGY_CHIER_RECTANGLEHELIO_CUH

#include "Heliostat.cuh"

class RectangleHelio : public Heliostat{
public:
    RectangleHelio(){};

    virtual int getSubHelioSize(){
        return sub_helio_size_;
    }

    virtual void setSize(float3 size_);

    virtual int CGetDiscreteMicroHelioOriginsAndNormals(float3 *&d_microhelio_vertexes, float3 *&d_microhelio_normals);

    virtual void CGetSubHeliostatVertexes(std::vector<float3> &);

    virtual void setSurfaceProperty(const std::vector<float> &surface_property){}
    virtual std::vector<float> getSurfaceProperty();

};

#endif //SOLARENERGY_CHIER_RECTANGLEHELIO_CUH