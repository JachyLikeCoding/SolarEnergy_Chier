//
// Created by feng on 19-4-16.
//

#ifndef SOLARENERGY_CHIER_NEWTYPEHELIOSTATFORTEST_H
#define SOLARENERGY_CHIER_NEWTYPEHELIOSTATFORTEST_H

#include "Heliostat.cuh"
#include "global_function.cuh"


class NewTypeHeliostatForTest : public Heliostat {
public:
    virtual int CGetDiscreteMicroHelioOriginsAndNormals(float3 *&d_microhelio_vertexs, float3 *&d_microhelio_normals) {}

    virtual int getSubHelioSize();
    virtual void setSize(float3 size);

    virtual void CGetSubHeliostatVertexes(std::vector<float3> &SubHeliostatVertexes);

    virtual void setSurfaceProperty(const std::vector<float> &surface_property){}
    virtual std::vector<float> getSurfaceProperty();
};



#endif //SOLARENERGY_CHIER_NEWTYPEHELIOSTATFORTEST_H
