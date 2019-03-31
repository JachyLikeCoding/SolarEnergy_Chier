//
// Created by feng on 19-3-31.
// PS: Define the rectangle receiver data structure.
//

#include "RectangleReceiver.cuh"

RectangleReceiver::RectangleReceiver(const RectangleReceiver &rectangle_receiver) : Receiver(rectangle_receiver){
    for(int i = 0; i < 4; ++i){
        rect_vertexes_[i] = rectangle_receiver.getRectangleVertex(i);
    }
    local_normal_ = rectangle_receiver.getLocalNormal();

}

void RectangleReceiver::CInit(int pixel_per_meter_for_receiver){
    pixel_length_ = 1.0f / float(pixel_per_meter_for_receiver);
    Cinit_vertex();
    setFocusCenter();
    Cset_resolution(pixel_per_meter_for_receiver);
    Calloc_image();     // cudaMalloc according to resolution. size = resulotion.x * resolution.y
    Cclean_image_content();
}


float3 RectangleReceiver::getRectangleVertex(int index) const {
    if(index > 3){
        throw std::runtime_error("The index should between [0, 3].");
    }
    return rect_vertexes_[index];
}

float3 RectangleReceiver::getLocalNormal() const {
    return local_normal_;
}

float3 RectangleReceiver::getFocusCenter(const float3 &heliostat_position) {
    return focus_center_;
}

void RectangleReceiver::Cinit_vertex(){
    Cset_local_normal();           //Set local normal
    Cset_local_vertex();           //Set local vertex according to face type
    Cset_world_vertex();           //Set world vertex according to normal
}

void RectangleReceiver::Cset_local_normal(){
    switch(surface_index_){
        case 0: local_normal_ = make_float3(0.0f, 0.0f, 1.0f); break;
        case 1: local_normal_ = make_float3(1.0f, 0.0f, 0.0f); break;
        case 2: local_normal_ = make_float3(0.0f, 0.0f, -1.0f); break;
        case 3: local_normal_ = make_float3(-1.0f, 0.0f, 0.0f); break;
        default: break;
    }
}

void RectangleReceiver::Cset_local_vertex(){
    switch(surface_index_){
        case 0:
            rect_vertexes_[0] = make_float3(-size_.x / 2, -size_.y / 2, size_.z / 2);
            rect_vertexes_[1] = make_float3(-size_.x / 2,  size_.y / 2, size_.z / 2);
            rect_vertexes_[2] = make_float3( size_.x / 2,  size_.y / 2, size_.z / 2);
            rect_vertexes_[3] = make_float3( size_.x / 2, -size_.y / 2, size_.z / 2);
            break;
        case 1:
            rect_vertexes_[0] = make_float3(size_.x / 2, -size_.y / 2,  size_.z / 2);
            rect_vertexes_[1] = make_float3(size_.x / 2,  size_.y / 2,  size_.z / 2);
            rect_vertexes_[2] = make_float3(size_.x / 2,  size_.y / 2, -size_.z / 2);
            rect_vertexes_[3] = make_float3(size_.x / 2, -size_.y / 2, -size_.z / 2);
            break;
        case 2:
            rect_vertexes_[0] = make_float3( size_.x / 2, -size_.y / 2, -size_.z / 2);
            rect_vertexes_[1] = make_float3( size_.x / 2,  size_.y / 2, -size_.z / 2);
            rect_vertexes_[2] = make_float3(-size_.x / 2,  size_.y / 2, -size_.z / 2);
            rect_vertexes_[3] = make_float3(-size_.x / 2, -size_.y / 2, -size_.z / 2);
            break;
        case 3:
            rect_vertexes_[0] = make_float3(-size_.x / 2, -size_.y / 2, -size_.z / 2);
            rect_vertexes_[1] = make_float3(-size_.x / 2,  size_.y / 2, -size_.z / 2);
            rect_vertexes_[2] = make_float3(-size_.x / 2,  size_.y / 2,  size_.z / 2);
            rect_vertexes_[3] = make_float3(-size_.x / 2, -size_.y / 2,  size_.z / 2);
            break;
        default:
            break;
    }
}

void RectangleReceiver::Cset_world_vertex(){
        normal_ = normalize(normal_);
        for(float3 vertex : rect_vertexes_){
            vertex = global_func::rotateY(vertex, local_normal_, normal_);
            vertex = global_func::translate(vertex, pos_);
        }
}

void RectangleReceiver::Cset_resolution(int pixel_per_meter_for_receiver){
    resolution_.x = int(size_.x * float(pixel_per_meter_for_receiver));
    resolution_.y = int(size_.y * float(pixel_per_meter_for_receiver));
}

void RectangleReceiver::setFocusCenter(){
    focus_center_ = (rect_vertexes_[0] + rect_vertexes_[2]) / 2;
}