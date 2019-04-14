//
// Created by feng on 19-4-11.
//

#ifndef SOLARENERGY_CHIER_SCENEPROCESSOR_H
#define SOLARENERGY_CHIER_SCENEPROCESSOR_H

#include "SceneConfiguration.h"
#include "SolarScene.h"

class SceneProcessor{
public:
    SceneProcessor() : sceneConfiguration(nullptr) {}

    SceneProcessor(SceneConfiguration *sceneConfigure) : sceneConfiguration(sceneConfigure){}

    bool processScene(SolarScene *solarScene);

    bool set_grid_receiver_heliostat_content(std::vector<Grid *> &grids,
                                         std::vector<Receiver *> &receivers,
                                         std::vector<Heliostat *> &heliostats);

    bool set_sunray_content(Sunray &sunray);

    SceneConfiguration *getSceneConfiguration() const;

    void setSceneConfiguration(SceneConfiguration *sceneConfiguration);


private:
    SceneConfiguration *sceneConfiguration;
    bool set_perturbation(Sunray &sunray);
    bool set_samplelights(Sunray &sunray);

};






















#endif //SOLARENERGY_CHIER_SCENEPROCESSOR_H
