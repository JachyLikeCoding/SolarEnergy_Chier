//
// Created by feng on 19-4-15.
//

#include "check_cuda.h"
#include "cuda_runtime.h"
#include "global_function.cuh"
#include "QuasiMonteCarloRayTracer.h"
#include "RectangleReceiverRectangleGridRayTracing.cuh"
#include "RectangleGrid.cuh"



void QuasiMonteCarloRayTracer::rayTracing(SolarScene *solarScene, int heliostat_id){
    /**
     * check valid heliostat index
     */
    checkValidHeliostatIndex(solarScene, heliostat_id);

    /**
     * Data Structure
     */
    //  -sunray
    Sunray *sunray = solarScene->getSunray();
    //  -heliostat
    Heliostat *heliostat = solarScene->getHeliostats()[heliostat_id];
    //  -grid
    int grid_id = heliostat->getBelongingGridId();
    Grid *grid = solarScene->getGrids()[grid_id];
    //  -receiver
    int receiver_id = grid->getBelongingReceiverIndex();
    Receiver *receiver = solarScene->getReceivers()[receiver_id];

    /**
     * Construct the sub_heliostat vertexes array
     */
    float3 *d_subHeliostat_vertexes = nullptr;
    setFlatRectangleHeliostatVertexs(d_subHeliostat_vertexes, solarScene->getHeliostats(),
                                     grid->getStartHeliostatIndex(),
                                     grid->getStartHeliostatIndex() + grid->getNumberOfHeliostats());

    /**
     *  Construct arguments
     */
     SunrayArgument sunrayArgument = generateSunrayArgument(sunray);
     HeliostatArgument heliostatArgument = generateHeliostatArgument(solarScene, heliostat_id);

     int receiverGridCombinationIndex = receiverGridCombination(receiver->getType(), grid->getGridType());
     //receiverGridCombinationIndex = receiver_type * 10 + grid_type;
     switch(receiverGridCombinationIndex){
         case 0:{
             /** RectangleReceiver & RectangleGrid*/
             auto rectangleReceiver = dynamic_cast<RectangleReceiver *>(receiver);
             auto rectangleGrid = dynamic_cast<RectangleGrid *>(grid);

             float ratio = heliostat->getPixelLength() / receiver->getPixelLength();
             float factor = sunray->getDNI() * ratio * ratio * sunray->getReflectiveRate()
                            / float(sunray->getNumOfSunshapeLightsPerGroup());
             RectangleReceiverRectangleGridRayTracing(sunrayArgument, rectangleReceiver, rectangleGrid,
                                                        heliostatArgument, d_subHeliostat_vertexes, factor);

             break;
         }
         case 10:{
            /** CylinderReceiver & RectangleGrid*/

             break;
         }

         /**
          * TODO: Add other branch for different type of receiver or grid.
          */

         default:
             break;
     }

     cudaFree(d_subHeliostat_vertexes);
     d_subHeliostat_vertexes = nullptr;
     heliostatArgument.CClear();
}

/**
 *  Public following function just for test. Do not use them standalone.
 */
void QuasiMonteCarloRayTracer::checkValidHeliostatIndex(SolarScene *solarScene, int heliostat_id){
    std::string error_message;
    std::string suffix = " is invalid.";

    // 1. Valid heliostat id
    error_message = "The ray tracing heliostat index " + std::to_string(heliostat_id);
    size_t  total_heliostat = solarScene->getHeliostats().size();
    if(heliostat_id < 0 || heliostat_id >= total_heliostat){
        std::string total_heliostat_message = "The heliostat index should between 0 and " + std::to_string(total_heliostat) + ".";
        throw std::runtime_error(error_message + suffix + total_heliostat_message);
    }
    Heliostat *heliostat = solarScene->getHeliostats()[heliostat_id];

    // 2. Valid grid id
    int grid_id = heliostat->getBelongingGridId();
    error_message += "with belonging grid index " + std::to_string(grid_id);
    size_t total_grid = solarScene->getGrids().size();
    if(grid_id < 0 || grid_id >= total_grid){
        std::string total_grid_message = "The grid index should between 0 and " + std::to_string(total_grid) + ".";
        throw std::runtime_error(error_message + suffix + total_grid_message);
    }
    Grid *grid = solarScene->getGrids()[grid_id];

    // 3. Valid receiver id
    int receiver_id = grid->getBelongingReceiverIndex();
    error_message += "of belonging receiver index " + std::to_string(receiver_id);
    size_t total_receivers = solarScene->getReceivers().size();
    if(receiver_id < 0 || receiver_id >= total_receivers){
        std::string total_receiver_message = "The receiver index should between 0 and " + std::to_string(total_receivers) + ".";
        throw std::runtime_error(error_message + suffix + total_receiver_message);
    }
}


int QuasiMonteCarloRayTracer::receiverGridCombination(int receiver_type, int grid_type){
    return receiver_type * 10 + grid_type;
}


HeliostatArgument QuasiMonteCarloRayTracer::generateHeliostatArgument(SolarScene *solarScene, int heliostat_id){
    Heliostat *heliostat = solarScene->getHeliostats()[heliostat_id];

    float3 *d_microhelio_origins = nullptr;
    float3 *d_microhelio_normals = nullptr;

    int numberOfMicrohelio = heliostat->CGetDiscreteMicroHelioOriginsAndNormals(d_microhelio_origins, d_microhelio_normals);

    int pool_size = solarScene->getSunray()->getNumOfSunshapeGroups() *
                    solarScene->getSunray()->getNumOfSunshapeLightsPerGroup();

    int *d_microhelio_belonging_groups = heliostat->generateDeviceMicrohelioGroup(pool_size, numberOfMicrohelio);

    int subHeliostat_id = 0;

    Grid *grid = solarScene->getGrids()[heliostat->getBelongingGridId()];
    for(int i = 0; i < grid->getNumberOfHeliostats(); ++i){
        int real_id = i + grid->getStartHeliostatIndex();
        if(real_id == heliostat_id)
            break;

        Heliostat *before_heliostat = solarScene->getHeliostats()[real_id];
        subHeliostat_id += before_heliostat->getSubHelioSize();
    }
    return HeliostatArgument(d_microhelio_origins, d_microhelio_normals, d_microhelio_belonging_groups,
                                numberOfMicrohelio, subHeliostat_id, heliostat->getSubHelioSize());
}


SunrayArgument QuasiMonteCarloRayTracer::generateSunrayArgument(Sunray *sunray){
    // SunrayArgument(float3 *samplelights, float3 *perturbations, int pool_size_, int lightsPerGroup, float3 sunray_dir)
    return SunrayArgument(sunray->getDeviceSampleLights(),          //float3 *samplelights
                          sunray->getDevicePerturbation(),          //float3 *perturbations
                          sunray->getNumOfSunshapeGroups() * sunray->getNumOfSunshapeLightsPerGroup(),    //pool_size_
                          sunray->getNumOfSunshapeLightsPerGroup(), //int lightsPerGroup
                          sunray->getSunDirection());               //float3 sunray_dir
}


/**
 *
 * @param d_heliostat_vertexs : SubHeliostat vertexes
 * @param heliostats
 * @param start_id
 * @param end_id
 * @return the number of subHeliostats
 */
int QuasiMonteCarloRayTracer::setFlatRectangleHeliostatVertexs(float3 *&d_heliostat_vertexs, std::vector<Heliostat *> &heliostats,
                                                                int start_id, int end_id){
    if(start_id < 0 || start_id > end_id || end_id > heliostats.size()){
        throw std::runtime_error(__FILE__ ". The index " + std::to_string(start_id) + " and " + std::to_string(end_id) + " is invalid.");
    }

    // clean up d_heliostat_vertexes
    if(d_heliostat_vertexs != nullptr){
        checkCudaErrors(cudaFree(d_heliostat_vertexs));
        d_heliostat_vertexs = nullptr;
    }

    std::vector<float3> subHeliostatVertexes;
    for(int i = start_id; i < end_id; ++i){
        heliostats[i]->CGetSubHeliostatVertexes(subHeliostatVertexes);
    }

    auto *h_subHeliostat_vertexes = new float3[subHeliostatVertexes.size()];
    for(int i = 0; i < subHeliostatVertexes.size(); ++i){
        h_subHeliostat_vertexes[i] = subHeliostatVertexes[i];
    }

    global_func::cpu2gpu(d_heliostat_vertexs, h_subHeliostat_vertexes, subHeliostatVertexes.size());
    delete[] h_subHeliostat_vertexes;
    h_subHeliostat_vertexes = nullptr;

    return int(subHeliostatVertexes.size());
}

