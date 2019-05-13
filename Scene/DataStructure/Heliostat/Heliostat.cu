//
// Created by feng on 19-3-28.
// PS: Define the heliostat data structure.
//

#include "Heliostat.cuh"
#include "global_function.cuh"
#include "RandomNumberGenerator/RandomGenerator.cuh"

/**
 * Calculate the normal of the heliostat.
 * @param focus_center : the center of the belonging receiver
 * @param sunray_dir : the sun ray direction after normalized
 */
void Heliostat::CSetNormal(const float3 &focus_center, const float3 &sunray_dir){
    float3 local_center = make_float3(pos_.x, pos_.y, pos_.z);
    float3 reflect_dir = focus_center - local_center;
    reflect_dir = normalize(reflect_dir);
    normal_ = normalize(reflect_dir - sunray_dir);
    printf("\nnormal：%f,%f,%f\n",normal_.x,normal_.y,normal_.z);
}

/**
 * Transform the local vertexs to world coordinate system.
 */
void Heliostat::CSetWorldVertex() {
    vertex_[0] = make_float3(-size_.x/2, size_.y/2, -size_.z/2);
    vertex_[1] = make_float3(0, 0, size_.z) + vertex_[0];
    vertex_[2] = make_float3(size_.x, 0, size_.z) + vertex_[0];
    vertex_[3] = make_float3(size_.x, 0, 0) + vertex_[0];
    //First step: rotate
    vertex_[0] = global_func::local2world_rotate(vertex_[0], normal_);
    vertex_[1] = global_func::local2world_rotate(vertex_[1], normal_);
    vertex_[2] = global_func::local2world_rotate(vertex_[2], normal_);
    vertex_[3] = global_func::local2world_rotate(vertex_[3], normal_);
    //Second step: translate
    vertex_[0] = global_func::translate(vertex_[0], pos_);
    vertex_[1] = global_func::translate(vertex_[1], pos_);
    vertex_[2] = global_func::translate(vertex_[2], pos_);
    vertex_[3] = global_func::translate(vertex_[3], pos_);
}


/**
 * 1) Set normal
 * 2)change vertexes from local to world
 * @param focus_center
 * @param sunray_dir
 */
void Heliostat::CSetNormalAndRotate(const float3 &focus_center, const float3 &sunray_dir) {
    CSetNormal(focus_center, sunray_dir);
    CSetWorldVertex();
}

/**
 *
 */
int* Heliostat::generateDeviceMicrohelioGroup(int num_group, int size) {
    int *d_groups = nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_groups, sizeof(int) * size));
    RandomGenerator::gpu_Uniform(d_groups, 0, num_group, size);
    return d_groups;
}


/**
* Getters and setters of attributes for heliostat object.
*/
float3 Heliostat::getPosition() const {
    return pos_;
}

void Heliostat::setPosition(float3 pos) {
    pos_ = pos;
}

float3 Heliostat::getSize() const {
    return size_;
}

float3 Heliostat::getNormal() const {
    return normal_;
}

void Heliostat::setNormal(float3 normal) {
    normal_ = normal;
}

int2 Heliostat::getRowAndColumn() const {
    return row_col_;
}

void Heliostat::setRowAndColumn(int2 row_col) {
    row_col_ = row_col;
}

float2 Heliostat::getGap() const {
    return gap_;
}

void Heliostat::setGap(float2 gap) {
    gap_ = gap;
}

SubCenterType Heliostat::getSubCenterType() const {
    return subCenterType_;
}

void Heliostat::setSubCenterType(SubCenterType type) {
    subCenterType_ = type;
}

float Heliostat::getPixelLength() const {
    return pixel_length_;
}

void Heliostat::setPixelLength(float pixel_length) {
    pixel_length_ = pixel_length;
}

int Heliostat::getBelongingGridId() const {
    return belonging_grid_id_;
}

void Heliostat::setBelongingGridId(int belonging_grid_id) {
    belonging_grid_id_ = belonging_grid_id;
}


std::vector<float3> Heliostat::getCPULocalNormals(){
    std::vector<float3> cpu_local_normals;
    for(int i = 0; i < row_col_.x * row_col_.y; ++i){
        cpu_local_normals.push_back(h_local_normals[i]);
    }
    return cpu_local_normals;
}

void Heliostat::setCPULocalNormals(float3 *h_local_normals_){
    h_local_normals = h_local_normals_;
}


void Heliostat::setCPULocalNormals(std::vector<float3> local_normals){
    int size = local_normals.size();
    h_local_normals = new float3[size];
    for (int i = 0; i < size; ++i) {
        h_local_normals[i] = local_normals[i];
    }
}

std::vector<float3> Heliostat::getCPULocalCenters(){
    std::vector<float3> cpu_local_centers;
    for(int i = 0; i < row_col_.x * row_col_.y; ++i){
//        printf("h_local_centers[%d] = (%f,%f,%f)\n",i,h_local_centers[i].x, h_local_centers[i].y, h_local_centers[i].z);
        cpu_local_centers.push_back(h_local_centers[i]);
    }
    return cpu_local_centers;
}



void Heliostat::setCPULocalCenters(float3 *h_local_centers_){
    h_local_centers = h_local_centers_;
}

void Heliostat::setCPULocalCenters(std::vector<float3> local_centers){
    int size = local_centers.size();
    h_local_centers = new float3[size];
    for (int i = 0; i < size; ++i) {
        h_local_centers[i] = local_centers[i];
    }
}