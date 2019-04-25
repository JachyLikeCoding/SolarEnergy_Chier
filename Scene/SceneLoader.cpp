//
// Created by feng on 19-4-9.
//
#include <iostream>
#include <fstream>

#include "SceneLoader.h"
#include "RectangleReceiver.cuh"
#include "CylinderReceiver.cuh"
#include "RectangleHelio.cuh"
#include "RectangleGrid.cuh"
#include "FocusFlatRectangleHelio.cuh"

/**TODO:
 * Add other types of components.
 */

int setIntField(const std::string &field_name, std::istream &stringstream){
    string head;
    stringstream >> head;

    if(field_name != head){
        throw std::runtime_error("Miss ' " + field_name + " ' property.\n");
    }

    int ans;
    stringstream >> ans;
    return ans;
}

float3 setFloat3Field(std::string field_name, std::istream &stringstream){
    string head;
    stringstream >> head;

    if(field_name != head){
        throw std::runtime_error("Miss ' " + field_name + " ' property.\n");
    }
    float3 ans;
    stringstream >> ans.x >> ans.y >> ans.z;
    return ans;
}

void checkIsolatedField(const std::string &field_name, std::istream &stringstream){
    string head;
    stringstream >> head;
    if(field_name != head){
        throw std::runtime_error("Miss ' " + field_name + " ' property.\n");
    }
}


class incorresponding_error{
public:
    incorresponding_error(std::string error_message){
        incorresponding_error::error_message = error_message;
    }

    incorresponding_error(const std::string &wrong_field_name, int expect_number, int real_number,
                          const std::string &belong_field_name, int belong_field_id){
        setErrorMessage(wrong_field_name, expect_number, real_number, belong_field_name, belong_field_id);
    }

    incorresponding_error(const std::string &wrong_field_name, int wrong_field_id, int wrong_belong_id,
                          const std::string &belong_field_name, int expect_belong_field_id1, int expect_belong_field_id2){
        setErrorMessage(wrong_field_name, wrong_field_id, wrong_belong_id, belong_field_name, expect_belong_field_id1, expect_belong_field_id2);
    }

    std::string what_error(){
        return error_message;
    }


private:
    std::string error_message;
    void setErrorMessage(const std::string &wrong_field_name, int expect_number, int real_number,
            const std::string &belong_field_name, int belong_field_id){
        error_message = "The " + std::to_string(belong_field_id) + "-th" + belong_field_name + " should contain " +\
        std::to_string(expect_number) + " " + wrong_field_name + "(s), but got " +
        std::to_string(real_number) + " " + wrong_field_name + "(s).";
    }

    void setErrorMessage(const std::string &wrong_field_name, int wrong_field_id, int real_belong_id,
            const std::string &belong_field_name, int expect_belong_field_id1, int expect_belong_field_id2){
            error_message =
                "The " + std::to_string(wrong_field_id) + "-th " + wrong_field_name + " is expected belonging to " +
                std::to_string(expect_belong_field_id1) + "-th ";
            if(expect_belong_field_id1 != expect_belong_field_id2){
                error_message += "or " + std::to_string(expect_belong_field_id2) + "-th";
            }
            error_message += belong_field_name + ", but got " + std::to_string(real_belong_id) + ".";
    }

};


void SceneLoader::add_ground(SolarScene *solarScene, std::istream &stringstream) {
    float ground_length, ground_width;
    stringstream >> ground_length >> ground_width;
    solarScene->setGroundLength(ground_length);
    solarScene->setGroundWidth(ground_width);
    solarScene->setGridNum(setIntField("ngrid", stringstream));
}

void SceneLoader::add_receiver(SolarScene *solarScene, std::istream &stringstream) {
    int type;
    stringstream >> type;

    Receiver *receiver = nullptr;
    switch(type){
        case 0:
            receiver = new RectangleReceiver();
            break;
        case 1:
            receiver = new CylinderReceiver();
            break;
        default:
            throw std::runtime_error("Receiver type has not defined.\n");
    }

    receiver->setType(type);
    receiver->setPosition(setFloat3Field("pos", stringstream));
    receiver->setSize(setFloat3Field("size", stringstream));
    receiver->setNormal(setFloat3Field("norm", stringstream));
    receiver->setSurfaceIndex(setIntField("face", stringstream));
    checkIsolatedField("end", stringstream);
    solarScene->addReceiver(receiver);
}


int SceneLoader::add_grid(SolarScene *solarScene, std::istream &stringstream, int receiver_index, int heliostat_start_index) {
    int type;
    stringstream >> type;

    Grid *grid = nullptr;

    switch (type){
        case 0:
            grid = new RectangleGrid();
            break;
        default:
            throw std::runtime_error("Grid type has not defined.\n");
    }

    grid->setGridType(type);
    grid->setBelongingReceiverIndex(receiver_index);
    grid->setStartHeliostatIndex(heliostat_start_index);

    grid->setPosition(setFloat3Field("pos", stringstream));
    grid->setSize(setFloat3Field("size", stringstream));
    grid->setInterval(setFloat3Field("inter", stringstream));
    grid->setNumberOfHeliostats(setIntField("n", stringstream));
    int heliostat_type = setIntField("type", stringstream);
    grid->setHeliostatType(heliostat_type);
    checkIsolatedField("end", stringstream);

    solarScene->addGrid(grid);
    return heliostat_type;

}

void SceneLoader::add_heliostat(SolarScene *solarScene, std::istream &stringstream,
        int type, float2 gap, int2 matrix, const std::vector<float> &surface_property,
        vector<float3> &h_local_centers_, vector<float3> &h_local_normals_) {
    Heliostat *heliostat = nullptr;
    switch(type){
        case 0:
            heliostat = new RectangleHelio();   // flat heliostat
            break;
        case 1:
            heliostat = new FocusFlatRectangleHelio();  // focus flat heliostat
            break;
        default:
            throw std::runtime_error("Heliostat has not defined.\n");
    }

    switch(int(surface_property[1])){
        case -1:
            heliostat->setCPULocalCenters(nullptr);
            heliostat->setCPULocalNormals(nullptr);
            break;
        case 1:
            heliostat->setCPULocalCenters(h_local_centers_);
            heliostat->setCPULocalNormals(h_local_normals_);
            break;
    }
    heliostat->setGap(gap);
    heliostat->setRowAndColumn(matrix);
    heliostat->setSurfaceProperty(surface_property);

    float3 position, size;
    stringstream >> position.x >> position.y >> position.z;
    stringstream >> size.x >> size.y >> size.z;
    heliostat->setPosition(position);
    heliostat->setSize(size);
    heliostat->setBelongingGridId(solarScene->getGrids().size()-1);

    solarScene->addHeliostat(heliostat);
}

void SceneLoader::checkScene(SolarScene *solarScene) {
    /**
     * Check whether the scene is valid:
     * - scene.grid_num_ == scene.grids.size()
     * - each grid in the scene:
     *      - belonging_receiver_index_ should be continuous increasing.
     *      - the num_helios == grid[i+1].start_helio_pos_ - grid[i],start_helio_pos_
     */

    if(solarScene->getGridNum() != solarScene->getGrids().size()){
        string error_message = "The scene should contains " + std::to_string(solarScene->getGridNum())
                + " grids, but got " + std::to_string(solarScene->getGrids().size()) + " grids.";
        throw incorresponding_error(error_message);
    }

    vector<Grid *> grids = solarScene->getGrids();
    for(int i = 0; i < grids.size(); ++i){
        int num_of_expect_heliostat = grids[i]->getNumberOfHeliostats();
        int num_of_exist_heliostat = (i == grids.size() - 1) ?
                solarScene->getHeliostats().size() : grids[i+1]->getStartHeliostatIndex();
        num_of_exist_heliostat -= grids[i]->getStartHeliostatIndex();

        if(num_of_exist_heliostat != num_of_expect_heliostat){
            throw incorresponding_error("heliostat", num_of_expect_heliostat, num_of_exist_heliostat, "grid", i+1);
        }

        int belong_receiver_index = grids[i]->getBelongingReceiverIndex();

        if(!i && belong_receiver_index != 0){
            throw incorresponding_error("grid", i, belong_receiver_index, "receiver", 0, 0);
        }else if(i){
            int prev_belong_receiver_index = grids[i-1]->getBelongingReceiverIndex();
            if(belong_receiver_index != prev_belong_receiver_index &&
               belong_receiver_index != prev_belong_receiver_index + 1){
                throw incorresponding_error("grid", i, belong_receiver_index, "receiver", prev_belong_receiver_index, prev_belong_receiver_index+1);
            }
        }
    }
}




bool SceneLoader::SceneFileRead(SolarScene *solarScene, std::string filepath) {
    int2 matrix = make_int2(1,1);
    float2 gap = make_float2(0.0f, 0.0f);
    std::vector<float> surface_property(6, -1.0f);

    std::vector<float3> local_centers;//===============================================add here
    std::vector<float3> local_normals;

    std::string head;
    int receiver_id = -1;
    int grid_id_for_each_receiver = 0;
    int heliostat_id_for_each_grid = 0;
    int current_total_heliostat = 0;

    int current_total_local_centers = 0;    //============================================add here
    int current_total_local_normals = 0;
    int current_total_subheliostat = 0;

    int heliostat_type = -1;

    try{
        std::ifstream scene_file(filepath);
        if(scene_file.fail()){
            throw std::runtime_error("Cannot open the file: " + filepath);
        }
        stringstream scene_stream;
        scene_stream << scene_file.rdbuf();
        scene_file.close();

        current_status = sceneRETree_.getRoot();
        while(scene_stream >> head){
            if(head[0] == '#'){
                std::string comment;
                getline(scene_stream, comment);
                continue;
            }

            if(head == "gap"){
                scene_stream >> gap.x >> gap.y;
            }else if(head == "matrix"){
                scene_stream >> matrix.x >> matrix.y;
            }else if(head == "surface_property"){
                scene_stream >> surface_property[0] >> surface_property[1] >> surface_property[2]
                >> surface_property[3] >> surface_property[4] >> surface_property[5];
            }else if(head == "ground"){
                current_status = sceneRETree_.step_forward(current_status, 'D');
                add_ground(solarScene, scene_stream);
            }else if(head == "Recv"){
                current_status = sceneRETree_.step_forward(current_status, 'R');
                add_receiver(solarScene, scene_stream);
                ++receiver_id;
                grid_id_for_each_receiver = 0;
            }else if(head == "Grid"){
                current_status = sceneRETree_.step_forward(current_status, 'G');
                heliostat_type = add_grid(solarScene, scene_stream, receiver_id, current_total_heliostat);
                ++grid_id_for_each_receiver;
                heliostat_id_for_each_grid = 0;
            }else if(head == "helio"){
                current_status = sceneRETree_.step_forward(current_status, 'H');
                add_heliostat(solarScene, scene_stream, heliostat_type, gap, matrix, surface_property, local_centers, local_normals);
                ++heliostat_id_for_each_grid;
                ++current_total_heliostat;
            }else if(head == "subhelio"){
                std::string comment;
                getline(scene_stream, comment);

                for(; current_total_local_centers < matrix.x * matrix.y; ++current_total_local_centers){
                    float a,b,c;
                    scene_stream >> a >> b >> c;
                    local_centers.push_back(make_float3(a,b,c));
                    getline(scene_stream,comment);
                    ++current_total_heliostat;
                }

                for(; current_total_local_normals < matrix.x * matrix.y; ++current_total_local_normals){
                    float a,b,c;
                    scene_stream >> a >> b >> c;
                    local_normals.push_back(make_float3(a,b,c));
                    getline(scene_stream,comment);
                    ++current_total_heliostat;
                }

                if(current_total_heliostat / 2 != matrix.x * matrix.y){
                    std::cerr << "Error caused by wrong subhelio centers or normals. Please check your input files.\n";
                }
            }else{
                current_status = sceneRETree_.step_forward(current_status, '?');
            }
        }

        checkScene(solarScene);
        solarScene->SetLoaded_from_file(true);

        return true;
    }catch(incorresponding_error e){
        std::cerr << "Error caused by ' " << e.what_error() << " '." << endl;
        return false;
    }catch(std::runtime_error runtime_error1){
        std::cerr << "Error occurs at " << std::to_string(receiver_id + 1) << "-th receiver, "
                    << std::to_string(grid_id_for_each_receiver) << "-th grid, "
                    << std::to_string(heliostat_id_for_each_grid) << "-th heliostat with head '" << head
                    << "'.\nThis is caused by ' " << runtime_error1.what() << "'" << std::endl;
        return false;
    }
}
