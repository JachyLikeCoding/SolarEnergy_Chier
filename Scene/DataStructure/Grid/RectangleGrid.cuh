//
// Created by feng on 19-3-29.
// PS: Define the rectangle grid data structure.
//
#ifndef SOLARENERGY_CHIER_RECTGRID_CUH
#define SOLARENERGY_CHIER_RECTGRID_CUH

#include "Grid.cuh"
using namespace std;

class RectangleGrid : public Grid{
public:

    virtual void Cinit();
    virtual void CClear();
    //Set *d_grid_helio_match_, *d_grid_helio_index_ and num_grid_helio_match_
    virtual int CGridHelioMatch(const vector<Heliostat *> &h_helios);

    __device__ __host__ RectangleGrid() : d_grid_helio_index_(nullptr), d_grid_helio_match_(nullptr){}

    __device__ __host__ ~RectangleGrid(){
        if(d_grid_helio_match_) d_grid_helio_match_ = nullptr;
        if(d_grid_helio_index_) d_grid_helio_match_ = nullptr;
    }

    __device__ __host__ int3 getSubGridNumber() const{
        return subgrid_num_;
    }

    void setGridNumber(int3 grid_num_);

    __device__ __host__ int *getDeviceGridHeliostatMatch() const{
        return d_grid_helio_match_;
    }

    void setDeviceGridHeliostatMatch(int *d_grid_helio_match_);

    __device__ __host__ int *getDeviceGridHeliostatIndex() const{
        return d_grid_helio_index_;
    }

    void setDeviceGridHeliostatIndex(int *d_grid_helio_index_);

    size_t getNumberOfGridHeliostatMatch() const;
    void setNumberOfGridHeliostatMatch(size_t num_grid_helio_match_);


private:
    int3 subgrid_num_;                     // consists of x * y * z sub-grids.
    int *d_grid_helio_match_;           // size = num_grid_helio_match
    int *d_grid_helio_index_;           // size = size.x * size.y * size.z + 1
    size_t num_grid_helio_match_;
};

#endif //SOLARENERGY_CHIER_RECTGRID_CUH