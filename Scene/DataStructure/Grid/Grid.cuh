//
// Created by feng on 19-3-28.
// PS: Define the grid data structure.
//

#ifndef SOLARENERGY_CHIER_GRID_CUH
#define SOLARENERGY_CHIER_GRID_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "Heliostat.cuh"


class Grid{
public:
    __device__ __host__ Grid(){}
    virtual void Cinit() = 0;
    virtual void CClear() = 0;
    //Set *d_grid_helio_match_, *d_grid_helio_index_ and num_grid_helio_match_
    virtual int CGridHelioMatch(const vector<Heliostat *> &h_helios) = 0;

    /**
     * Getters and setters of attributes for Grid object.
     */
    int getGridType() const;
    void setGridType(int type_);

    __device__ __host__ float3 getPosition() const;
    void setPosition(float3 pos_);

    float3 getSize() const;
    void setSize(float3 size_);

    __device__ __host__ float3 getInterval() const;
    void setInterval(float3 interval_);

    int getHeliostatType() const;
    void setHeliostatType(int helio_type_);

    int getStartHeliostatIndex() const;
    void setStartHeliostatIndex(int start_helio_index_);

    int getNumberOfHeliostats() const;
    void setNumberOfHeliostats(int num_helios_);

    int getBelongingReceiverIndex() const;
    void setBelongingReceiverIndex(int belonging_receiver_index_);


protected:
    int type_;
    float3 pos_;
    float3 size_;
    float3 interval_;
    int helio_type_;
    int start_helio_index_;     //The first heliostat index of the heliostat lists in this grid
    int num_helios_;            //The number of heliostats in this grid
    int belonging_receiver_index_;

};


#endif //SOLARENERGY_CHIER_GRID_CUH