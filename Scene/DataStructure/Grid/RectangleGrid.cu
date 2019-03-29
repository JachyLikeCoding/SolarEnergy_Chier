//
// Created by feng on 19-3-29.
// PS: Define the rectangle grid data structure.
//
#include "RectangleGrid.cuh"
#include "check_cuda.h"
#include "global_function.cuh"

void RectangleGrid::Cinit(){
    grid_num_.x = int(ceil(size_.x / interval_.x));
    grid_num_.y = int(ceil(size_.y / interval_.y));
    grid_num_.z = int(ceil(size_.z / interval_.z));
}

void RectangleGrid::CClear(){
    if(d_grid_helio_index_){
        checkCudaErrors(cudaFree(d_grid_helio_index_));
        d_grid_helio_index_ = nullptr;
    }

    if(d_grid_helio_match_){
        checkCudaErrors(cudaFree(d_grid_helio_match_));
        d_grid_helio_match_ = nullptr;
    }
}


int boxIntersect(int helioBeginId, int subHelioSize, float3 minPos, float3 maxPos,
        const RectangleGrid &grid, vector<vector<int> > &grid_helio_match_vector){
    int size = 0;
    float3 pos = grid.getPosition();
    float3 interval = grid.getInterval();
    int3 subgrid_num = grid.getSubGridNumber();

    int3 min_grid_pos = make_int3((min_pos - pos) / interval);                      //===>don't forget to test whether it is right or not
    int3 max_grid_pos = make_int3((max_pos - pos) / interval);

    if(min_grid_pos.x < 0 || min_grid_pos.y < 0 || min_grid_pos.z < 0
        || max_grid_pos.x >= subgrid_num.x || max_grid_pos.y >= subgrid_num.y || max_grid_pos.z >= subgrid_num.z){
        std::cerr << "The heliostats may out of the grid boundary. Please check your input file." << std::endl;
    }

    for(int x = max(0 , min_grid_pos); x <= min(subgrid_num.x - 1 , max_grid_pos.x); ++x){
        for(int y = max(0, min_grid_pos.y); y <= min(subgrid_num.y - 1, max_grid_pos.y); ++y){
            for(int z = max(0, min_grid_pos.z); z <= min(subgrid_num.z - 1, max_grid_pos.z); ++z){
                int pos_id = x * subgrid_num.y * subgrid_num.z + y * subgrid_num.z + z;
                for(int i = 0; i < subHelioSize; ++i){
                    grid_helio_match_vector[pos_id].push_back(helioBeginId + i);
                }
                size += subHelioSize;
            }
        }
    }
    return size;
}


//Set *d_grid_helio_match_, *d_grid_helio_index_ and num_grid_helio_match_
int RectangleGrid::CGridHelioMatch(const vector<Heliostat *> &h_helios){
    if(d_grid_helio_match_ || d_grid_helio_index_){
        throw std::runtime_error("The grid and heliostats corresponding relationship should be empty before calling this method.");
    }

    int start_subhelio_pos = 0;
    float3 minPos, maxPos;
    float radius = 0.0f;
    num_grid_helio_match_ = 0;

    vector<vector<int> > grid_helio_match_vector(make_int1(grid_num_.x * grid_num_.y * grid_num_.z), vector<int>());
    for(int i = start_helio_index_; i < start_helio_index_ + num_helios_; ++i){
        float3 pos = h_helios[i]->getPosition();
        radius = length(h_helios[i]->getSize())/2;

        minPos = pos - radius;
        maxPos = pos + radius;

        num_grid_helio_match_ += boxIntersect(start_subhelio_pos, h_helios[i]->getSubCenterSize(), minPos, maxPos, *this, grid_helio_match_vector);
        start_subhelio_pos += h_helios[i]->getSubHelioSize();
    }

    int *h_grid_helio_index = new int[grid_num_.x * grid_num_.y * grid_num_.z + 1];
    h_grid_helio_index[0] = 0;

    int *h_grid_helio_match = new int[num_grid_helio_match_];

    int index = 0;
    for(int i = 0; i < grid_num_.x * grid_num_.y * grid_num_.z; ++i){
        h_grid_helio_index[i + 1] = int(h_grid_helio_index[i] + grid_helio_match_vector[i].size());

        for(int j = 0; j < grid_helio_match_vector[i].size(); ++j, ++index){
            h_grid_helio_match[index] = grid_helio_match_vector[i][j];
        }
    }

    global_func::cpu2gpu(d_grid_helio_index_, h_grid_helio_index, make_int1(grid_num_.x * grid_num_.y * grid_num_.z) + 1);
    global_func::cpu2gpu(d_grid_helio_match_, h_grid_helio_match, num_grid_helio_match_);

    delete[] h_grid_helio_index;
    delete[] h_grid_helio_match;

    h_grid_helio_index = nullptr;
    h_grid_helio_match = nullptr;

    return start_subhelio_pos;
}


/**
 * Getter and setter of attributes for Rectangle Grid
 */
void RectangleGrid::setGridNumber(int3 grid_num){
    grid_num_ = grid_num;
}

void RectangleGrid::setDeviceGridHeliostatMatch(int *d_grid_helio_match){
    d_grid_helio_match_ = d_grid_helio_match;
}

void RectangleGrid::setDeviceGridHeliostatIndex(int *d_grid_helio_index){
    d_grid_helio_index_ = d_grid_helio_index;
}

size_t RectangleGrid::getNumberOfGridHeliostatMatch() const{
    return num_grid_helio_match_;
}

void RectangleGrid::setNumberOfGridHeliostatMatch(size_t num_grid_helio_match){
    num_grid_helio_match_ = num_grid_helio_match;
}
