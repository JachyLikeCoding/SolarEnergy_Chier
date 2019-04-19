//
// Created by feng on 19-4-12.
//

#include <stdexcept>
#include "SceneProcessor.h"

// Set sunray, grids, receivers, heliostats
bool SceneProcessor::processScene(SolarScene *solarScene){
    return set_sunray_content(*solarScene->getSunray()) && set_grid_receiver_heliostat_content(solarScene->getGrids(), solarScene->getReceivers(), solarScene->getHeliostats());
}

bool SceneProcessor::set_grid_receiver_heliostat_content(std::vector<Grid *> &grids,
                                         std::vector<Receiver *> &receivers,
                                         std::vector<Heliostat *> &heliostats){
    if(!sceneConfiguration){
        throw std::runtime_error("No scene configuration. Please load it first before process scene.");
    }

    int pixels_per_meter_for_receiver = int(1.0f / sceneConfiguration->getReceiver_pixel_length());
    float heliostat_pixel_length = sceneConfiguration->getHelio_pixel_length();
    float3 sun_direction = sceneConfiguration->getSun_dir();

    for(Grid *grid : grids){
        grid->Cinit();
        grid->CGridHelioMatch(heliostats);

        Receiver *receiver = receivers[grid->getBelongingReceiverIndex()];
        receiver->CInit(pixels_per_meter_for_receiver);

        for(int i = 0; i < grid->getNumberOfHeliostats(); ++i){
            int id = i + grid->getStartHeliostatIndex();
            float3 focus_center = receiver->getFocusCenter(heliostats[id]->getPosition());
            heliostats[id]->setPixelLength(heliostat_pixel_length);
            heliostats[id]->CSetNormalAndRotate(focus_center, sun_direction);
        }
    }
    return true;
}

bool SceneProcessor::set_sunray_content(Sunray &sunray){
    if(!sceneConfiguration){
        throw std::runtime_error("No scene configuration. Please load it first before process scene.");
    }

    sunray.setSunDirection(sceneConfiguration->getSun_dir());
    sunray.setCSR(sceneConfiguration->getCsr());
    sunray.setDNI(sceneConfiguration->getDni());
    sunray.setNumOfSunshapeGroups(sceneConfiguration->getNum_sunshape_groups());
    sunray.setNumOfSunshapeLightsPerGroup(sceneConfiguration->getNum_sunshape_lights_per_group());
    sunray.setReflectiveRate(sceneConfiguration->getReflected_rate());

    return set_perturbation(sunray) && set_samplelights(sunray);
}

SceneConfiguration *SceneProcessor::getSceneConfiguration() const{
    return sceneConfiguration;
}

void SceneProcessor::setSceneConfiguration(SceneConfiguration *sceneConfiguration){
    SceneProcessor::sceneConfiguration = sceneConfiguration;
}