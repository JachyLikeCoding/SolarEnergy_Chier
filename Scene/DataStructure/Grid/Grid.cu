#include "Grid.cuh"

/**
* Getters and setters of attributes for Grid object.
*/
int Grid::getGridType() const {
    return type_;
}

void Grid::setGridType(int type) {
    type_ = type;
}

float3 Grid::getPosition() const{
    return pos_;
}

void Grid::setPosition(float3 pos) {
    pos_ = pos;
}

float3 Grid::getSize() const {
    return size_;
}

void Grid::setSize(float3 size) {
    size_ = size;
}

float3 Grid::getInterval() const{
    return interval_;
}

void Grid::setInterval(float3 interval){
    interval_ = interval;
}

int Grid::getHeliostatType() const{
    return helio_type_;
}
void Grid::setHeliostatType(int helio_type){
    helio_type_ = helio_type;
}

int Grid::getStartHeliostatIndex() const{
    return start_helio_index_;
}

void Grid::setStartHeliostatIndex(int start_helio_index){
    start_helio_index_ = start_helio_index;
}

int Grid::getNumberOfHeliostats() const{
    return num_helios_;
}

void Grid::setNumberOfHeliostats(int num_helios){
    num_helios_ = num_helios;
}

int Grid::getBelongingReceiverIndex() const{
    return belonging_receiver_index_;
}

void Grid::setBelongingReceiverIndex(int belonging_receiver_index){
    belonging_receiver_index_ = belonging_receiver_index;
}