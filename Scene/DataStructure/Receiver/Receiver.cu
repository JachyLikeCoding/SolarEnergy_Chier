//
// Created by feng on 19-3-28.
// PS: Define the receiver data structure.
//

#include "Receiver.cuh"
#include "check_cuda.h"
#include "global_function.cuh"

/**
  * Allocate the final image matrix.
  */
void Receiver::Calloc_image(){
    checkCudaErrors(cudaMalloc((void **) &d_image_, sizeof(float) * resolution_.x * resolution_.y));
}

/**
 * Clean the final image matrix.
 */
void Receiver::Cclean_image_content(){
    int n_resolution = resolution_.x * resolution_.y;
    float *h_clean_receiver = new float[n_resolution];
    for(int i = 0; i < n_resolution; ++i){
        h_clean_receiver[i] = 0.0f;
    }

    //Clean screen
    global_func::cpu2gpu(d_image_, h_clean_receiver, n_resolution);

    delete[] h_clean_receiver;
    h_clean_receiver = nullptr;
}

void Receiver::CClear(){
    if(d_image_){
        checkCudaErrors(cudaFree(d_image_));
        d_image_ = nullptr;
    }
}

/**
* Getters and setters of attributes for receiver object.
*/
int Receiver::getType() const{
    return type_;
}

void Receiver::setType(int type){
    type_ = type;
}

float3 Receiver::getNormal() const{
    return normal_;
}

void Receiver::setNormal(float3 normal){
    normal_ = normal;
}

float3 Receiver::getPosition() const{
    return pos_;
}

void Receiver::setPosition(float3 pos){
    pos_ = pos;
}

float3 Receiver::getSize() const{
    return size_;
}

void Receiver::setSize(float3 size){
    size_ = size;
}

int Receiver::getSurfaceIndex() const{
    return surface_index_;
}

void Receiver::setSurfaceIndex(int surface_index){
    surface_index_ = surface_index;
}

float Receiver::getPixelLength() const{
    return pixel_length_;
}

float *Receiver::getDeviceImage() const{
    return d_image_;
}

int2 Receiver::getResolution() const{
    return resolution_;
}