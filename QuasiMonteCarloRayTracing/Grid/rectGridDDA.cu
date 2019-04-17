#include "rectGridDDA.cuh"
#include "global_function.cuh"

/**
 * Intersect with heliostats in this rectangle grid.
 */
__host__ __device__ bool rectGridDDA::intersect(const float3 &origin, const float3 &dir,
                                                const float3 *d_heliostat_vertexes,
                                                const int *d_grid_heliostat_match,
                                                int start_id, int end_id, int subheliostat_id, int numberOfSubHeliostat) {
    float t, u, v;
    for(int i = start_id; i < end_id; ++i){
        int subHeliostatIndex = d_grid_heliostat_match[i];
        if(subHeliostatIndex < subheliostat_id || subHeliostatIndex >= subheliostat_id + numberOfSubHeliostat){
            if(global_func::rayParallelogramIntersect(origin, dir,
                                                      d_heliostat_vertexes[3 * subHeliostatIndex + 1],
                                                      d_heliostat_vertexes[3 * subHeliostatIndex],
                                                      d_heliostat_vertexes[3 * subHeliostatIndex + 2],
                                                      t, u, v)){
                return true;
            }
        }
    }
    return false;
}


/**
 * 3D-DDA
 */
__host__ __device__ bool rectGridDDA::collision(const float3 &origin, const float3 &dir, const RectangleGrid &rectGrid,
                                                const float3 *d_subheliostat_vertexes,
                                                const HeliostatArgument &heliostatArgument) {
    /**
     *Step 1 - Initialization
     **/
    //  Step 1.1 Initial current position of origin in the scene.
    int3 pos = make_int3((origin - rectGrid.getPosition()) / rectGrid.getInterval());

    //  Step 1.2 StepX, StepY, StepZ
    int3 Step;
    Step.x = (dir.x >= 0) ? 1 : -1;
    Step.y = (dir.y >= 0) ? 1 : -1;
    Step.z = (dir.z >= 0) ? 1 : -1;

    //  Step 1.3 Initial tmaxX, tmaxY, tmaxZ
    float3 tMax;    // avoid divide 0
    // tMax = |pos ± interval - (origin-rectGrid.getPosition())| / dir
    tMax.x = absDivide( calTMax(dir.x, rectGrid.getInterval().x, pos.x, origin.x-rectGrid.getPosition().x) , dir.x );
    tMax.y = absDivide( calTMax(dir.y, rectGrid.getInterval().y, pos.y, origin.y-rectGrid.getPosition().y) , dir.y );
    tMax.z = absDivide( calTMax(dir.z, rectGrid.getInterval().z, pos.z, origin.z-rectGrid.getPosition().z) , dir.z );

    //  Step 1.4 Initial tDeltaX, tDeltaY, tDeltaZ
    float3 tDelta;
    tDelta.x = absDivide(rectGrid.getInterval().x, dir.x);
    tDelta.y = absDivide(rectGrid.getInterval().y, dir.y);
    tDelta.z = absDivide(rectGrid.getInterval().z, dir.z);

    /**
     * Step 2 - Intersection
     **/
    int3 grid_index = pos;
    int grid_address;

    while(grid_index.x >= 0 && grid_index.x < rectGrid.getSubGridNumber().x &&
          grid_index.y >= 0 && grid_index.y < rectGrid.getSubGridNumber().y &&
          grid_index.z >= 0 && grid_index.z < rectGrid.getSubGridNumber().z){

        grid_address = global_func::unroll_index(grid_index, rectGrid.getSubGridNumber());

        if(intersect(origin, dir, d_subheliostat_vertexes,
                    rectGrid.getDeviceGridHeliostatMatch(),
                    rectGrid.getDeviceGridHeliostatIndex()[grid_address],
                    rectGrid.getDeviceGridHeliostatIndex()[grid_address + 1],
                    heliostatArgument.subHeliostat_id,
                    heliostatArgument.numberOfSubHeliostats)){
            return true;
        }

        // next cell location
        bool xMinFlag, yMinFlag, zMinFlag;
        xMinFlag = yMinFlag = zMinFlag = false;

        if(less(tMax.x, tMax.y)){   // x < y
            if(less(tMax.x, tMax.z) || equal(tMax.x, tMax.z)){  // x <= z,  +x minimal
                grid_index.x += Step.x;
                xMinFlag = true;
            }
            if(less(tMax.z, tMax.x) || equal(tMax.x, tMax.z)){  // z <= x,  +z minimal
                grid_index.z += Step.z;
                zMinFlag = true;
            }
        }else if(less(tMax.y, tMax.x)){ // y < x
            if(less(tMax.y, tMax.z) || equal(tMax.y, tMax.z)){  // y <= z,  +y minimal
                grid_index.y += Step.y;
                yMinFlag = true;
            }
            if(less(tMax.z, tMax.y) || equal(tMax.y, tMax.z)){  // z <= y,  +z minimal
                grid_index.z += Step.z;
                zMinFlag = true;
            }
        }else{  // x == y
            if(less(tMax.x, tMax.z) || equal(tMax.x, tMax.z)){  // x(y) <= z, +x&y minimal
                grid_index.x += Step.x;
                grid_index.y += Step.y;
                xMinFlag = yMinFlag = true;
            }
            if(less(tMax.z, tMax.x) || equal(tMax.x, tMax.z)){  // z <= x(y),   +z minimal
                grid_index.z += Step.z;
                zMinFlag = true;
            }
        }
        tMax.x += (xMinFlag) ? tDelta.x : 0;
        tMax.y += (yMinFlag) ? tDelta.y : 0;
        tMax.z += (zMinFlag) ? tDelta.z : 0;
    }

    return false;
}
