#include "CylinderReceiver.cuh"


void CylinderReceiver::CInit(int pixel_per_meter_for_receiver){
    pixel_length_ = 1.0f / float(pixel_per_meter_for_receiver);
    Cset_resolution(pixel_per_meter_for_receiver);
    Calloc_image();
    Cclean_image_content();
}


void CylinderReceiver::Cset_resolution(int pixel_per_meter_for_receiver){
    resolution_.x = int(ceil(2 * M_PI * size_.x * float(pixel_per_meter_for_receiver)));
    resolution_.y = int(size_.y * float(pixel_per_meter_for_receiver));
}


float3 CylinderReceiver::getFocusCenter(const float3 &heliostat_position){
    if(isInsideCylinder(heliostat_position)){
        // If the heliostat position is inside cylinder, return the receiver position
        return pos_;
    }

    float3 dir = normalize(heliostat_position - pos_);
    float radius = size_.x / (length(make_float2(dir.x, dir.z)));

    float x = pos_.x + dir.x * radius;
    float z = pos_.z + dir.z * radius;

    return make_float3(x, pos_.y, z);
}