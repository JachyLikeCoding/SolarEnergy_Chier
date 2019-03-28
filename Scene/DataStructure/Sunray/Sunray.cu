//
// Created by feng on 19-3-28.
// PS: Define the sun ray data structure.
//
#include "Sunray.cuh"
#include "check_cuda.h"


void Sunray::CClear() {
    if(d_perturbation_){
        checkCudaErrors(cudaFree(d_perturbation_));
        d_perturbation_ = nullptr;
    }
    if(d_samplelights_){
        checkCudaErrors(cudaFree(d_samplelights_));
        d_samplelights_ = nullptr;
    }
}

Sunray::~Sunray() {
    if(d_perturbation_)
        d_perturbation_ = nullptr;
    if(d_samplelights_)
        d_samplelights_ = nullptr;
}

/**
* Getters and setters of attributes for sun ray object.
*/
float3 Sunray::getSunDirection() const{
    return sun_dir_;
}
void Sunray::setSunDirection(float3 sun_dir){
    sun_dir_ = sun_dir;
}

float Sunray::getDNI() const{
    return dni_;
}
void Sunray::setDNI(float dni){
    dni_ = dni;
}

float Sunray::getCSR() const{
    return csr_;
}
void Sunray::setCSR(float csr){
    csr_ = csr;
}

int Sunray::getNumOfSunshapeGroups() const{
    return num_sunshape_groups_;
}
void Sunray::setNumOfSunshapeGroups(int num_sunshape_groups){
    num_sunshape_groups_ = num_sunshape_groups;
}

int Sunray::getNumOfSunshapeLightsPerGroup() const{
    return num_sunshape_lights_per_group_;
}
void Sunray::setNumOfSunshapeLightsPerGroup(int num_sunshape_lights_per_group){
    num_sunshape_lights_per_group_ = num_sunshape_lights_per_group;
}

float3 Sunray::getDeviceSampleLights() const{
    return d_samplelights_;
}
void Sunray::setDeviceSampleLights(float3 *d_samplelights){
    d_samplelights_ = d_samplelights;
}

float3 Sunray::getDevicePerturbation() const{
    return d_perturbation_;
}
void Sunray::setDevicePerturbation(float3 *d_perturbation){
    d_perturbation_ = d_perturbation;
}

float Sunray::getReflectiveRate() const{
    return reflective_rate_;
}
void Sunray::setReflectiveRate(float reflective_rate){
    reflective_rate_ = reflective_rate;
}