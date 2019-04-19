//
// Created by feng on 19-3-29.
//

#include "SolarScene.h"
#include "RandomNumberGenerator/RandomGenerator.cuh"
#include "destroy.h"

SolarScene* SolarScene::m_instance;
SolarScene* SolarScene::GetInstance() {
    if(m_instance == nullptr){
        m_instance = new SolarScene();
    }
    return m_instance;
}

SolarScene::SolarScene() : loaded_from_file_(false), sunray(nullptr) {
    //init the random generator
    RandomGenerator::initSeed();
    RandomGenerator::initCudaRandGenerator();

    // Allocate sunray
    sunray = new Sunray();
}

SolarScene::~SolarScene() {
    clear();
    m_instance = nullptr;
}

bool SolarScene::clear() {
    //1. Free memory on GPU
    free_scene::gpu_free(grids);
    free_scene::gpu_free(sunray);
    free_scene::gpu_free(receivers);    //free_scene : namespace in the "destroy.h"
    //2. Free memory on CPU
    free_scene::cpu_free(receivers);
    free_scene::cpu_free(grids);
    free_scene::cpu_free(heliostats);
    free_scene::cpu_free(sunray);

    //3. Clear vector
    receivers.clear();
    grids.clear();
    heliostats.clear();
}

/**
 * Add components in the solar scene.
 */
void SolarScene::addReceiver(Receiver *receiver){
    receivers.push_back(receiver);
}

void SolarScene::addHeliostat(Heliostat *heliostat){
    heliostats.push_back(heliostat);
}

void SolarScene::addGrid(Grid *grid){
    grids.push_back(grid);
}

/**
 * Get components in the solar scene.
 */
Sunray *SolarScene::getSunray(){
    return sunray;
}

vector<Grid *> &SolarScene::getGrids(){
    return grids;
}

vector<Receiver *> &SolarScene::getReceivers(){
    return receivers;
}

vector<Heliostat *> &SolarScene::getHeliostats(){
    return heliostats;
}

/**
* Getters and setters of attributes for SolarScene object.
*/
float SolarScene::getGroundLength() const{
    return ground_length_;
}

void SolarScene::setGroundLength(float ground_length_){
    SolarScene::ground_length_ = ground_length_;
}

float SolarScene::getGroundWidth() const{
    return ground_width_;
}

void SolarScene::setGroundWidth(float ground_width_){
    SolarScene::ground_width_ = ground_width_;
}

int SolarScene::getGridNum() const{
    return grid_num_;
}

void SolarScene::setGridNum(int grid_num_){
    SolarScene::grid_num_ = grid_num_;
}


bool SolarScene::getLoaded_from_file() const{
    return loaded_from_file_;
}

void SolarScene::SetLoaded_from_file(bool loaded_from_file){
    SolarScene::loaded_from_file_ = loaded_from_file;
}
