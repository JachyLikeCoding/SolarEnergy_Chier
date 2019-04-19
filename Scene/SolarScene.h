//
// Created by feng on 19-3-28.
//

#ifndef SOLARENERGY_CHIER_SOLARSCENE_H
#define SOLARENERGY_CHIER_SOLARSCENE_H

#include <vector>

#include "Heliostat.cuh"
#include "Grid.cuh"
#include "Receiver.cuh"
#include "Sunray.cuh"

using namespace std;

class SolarScene{
private:
    SolarScene();

    //Singleton
    static SolarScene *m_instance;
    bool loaded_from_file_;

    float ground_length_;
    float ground_width_;
    int grid_num_;

    Sunray *sunray;
    vector<Grid *> grids;
    vector<Heliostat *> heliostats;
    vector<Receiver *> receivers;

public:
    static SolarScene* GetInstance();
    ~SolarScene();
    bool clear();

    // Getters and setters of attributes for SolarScene object.
    float getGroundLength() const;
    void setGroundLength(float ground_length_);

    float getGroundWidth() const;
    void setGroundWidth(float ground_width_);

    int getGridNum() const;
    void setGridNum(int grid_num_);

    bool getLoaded_from_file() const;
    void SetLoaded_from_file(bool loaded_from_file_);

    // Add components in the solar scene.
    void addReceiver(Receiver *receiver);
    void addHeliostat(Heliostat *heliostat);
    void addGrid(Grid *grid);

    // Get components in the solar scene.
    Sunray *getSunray();
    vector<Grid *> &getGrids();
    vector<Receiver *> &getReceivers();
    vector<Heliostat *> &getHeliostats();
};

#endif //SOLARENERGY_CHIER_SOLARSCENE_H
