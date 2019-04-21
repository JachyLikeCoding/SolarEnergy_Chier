//
// Created by feng on 19-4-21.
//

#ifndef SOLARENERGY_CHIER_RECEIVERINTERSECTUTIL_H
#define SOLARENERGY_CHIER_RECEIVERINTERSECTUTIL_H

#include "SolarScene.h"
#include "global_function.cuh"

void changeSunLightAndPerturbationToParallel(Sunray *sunray);

std::vector<float> deviceArray2Vector(float *d_array, int size);

#endif //SOLARENERGY_CHIER_RECEIVERINTERSECTUTIL_H
