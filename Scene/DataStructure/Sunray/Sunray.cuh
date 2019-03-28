//
// Created by feng on 19-3-28.
// PS: Define the sun ray data structure.
//

#ifndef SOLARENERGY_CHIER_SUNRAY_CUH
#define SOLARENERGY_CHIER_SUNRAY_CUH

#include <cuda_runtime.h>

class Sunray{
public:
    __device__ __host__ Sunray() : d_samplelights_(nullptr), d_perturbation_(nullptr){}

    __device__ __host__ Sunray(float3 sun_dir, int num_sunshape_groups, int num_sunshape_lights_per_group, float dni, float csr) : Sunray(){
        sun_dir_ = sun_dir;
        dni_ = dni;
        csr_ = csr;
        num_sunshape_groups_ = num_sunshape_groups;
        num_sunshape_lights_per_group_ = num_sunshape_lights_per_group;
    }

    __device__ __host__ ~Sunray();

    void CClear();

    /**
     * Getters and setters of attributes for sun ray object.
     */
    __device__ __host__ float3 getSunDirection() const;
    void setSunDirection(float3 sun_dir_);

    __device__ __host__ float getDNI() const;
    void setDNI(float dni_);

    __device__ __host__ float getCSR() const;
    void setCSR(float csr_);

    __device__ __host__ int getNumOfSunshapeGroups() const;
    void setNumOfSunshapeGroups(int num_sunshape_groups_);

    __device__ __host__ int getNumOfSunshapeLightsPerGroup() const;
    void setNumOfSunshapeLightsPerGroup(int num_sunshape_lights_per_group_);

    __device__ __host__ float3 *getDeviceSampleLights() const;
    void setDeviceSampleLights(float3 *d_samplelights_);

    __device__ __host__ float3 *getDevicePerturbation() const;
    void setDevicePerturbation(float3 *d_perturbation_);

    float getReflectiveRate() const;
    void setReflectiveRate(float reflective_rate_);


private:
    float3 sun_dir_;                        //e.g.  0.306454 -0.790155   0.530793
    float dni_;                             //e.g.  1000.0
    float csr_;                             //e.g.  0.1
    int num_sunshape_groups_;               //e.g.  8
    int num_sunshape_lights_per_group_;     //e.g.  1024
    float reflective_rate_;                 //e.g.  0.88
                                            //Since all heliostats in the same scene with the same reflective rate in our system,
                                            // we put "reflective rate" in sunray datastructure.
                                            //One SolarScene only contains one sunray instance.
    float3 *d_samplelights_;                //point to sample lights memory on GPU
                                                //memory size = num_sunshape_groups_ * num_sunshape_lights_per_group_
                                                //e.g.        = 8 * 1024
    float3 *d_perturbation_;                //point to perturbation memory on GPU
                                                //memory size = num_sunshape_groups_ * num_sunshape_lights_per_group_
                                                //which obeys Gaussian distribution
};

#endif //SOLARENERGY_CHIER_SUNRAY_CUH