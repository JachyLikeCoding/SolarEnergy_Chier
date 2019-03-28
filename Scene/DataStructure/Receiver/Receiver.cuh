//
// Created by feng on 19-3-28.
// PS: Define the receiver data structure.
//

#ifndef SOLARENERGY_CHIER_RECEIVER_CUH
#define SOLARENERGY_CHIER_RECEIVER_CUH

#include <cuda_runtime.h>

class Receiver{
public:
    __device__ __host__ Receiver() : d_image_(nullptr){}

    __device__ __host__ Receiver(const Receiver &rec){
        type_ = rec.type_;
        normal_ = rec.normal_;
        pos_ = rec.pos_;
        size_ = rec.size_;
        surface_index_ = rec.surface_index_;
        pixel_length_ = rec.pixel_length_;
        d_image_ = rec.d_image_;
        resolution_ = rec.resolution_;
    }

    __device__ __host__ ~Receiver();

    /**
     * Initialize the parameters.
     */
    virtual void CInit(int geometry_info) = 0;
    virtual void Cset_resolution(int geometry_info) = 0;
    virtual float3 getFocusCenter(const float3 &heliostat_position) = 0;

    /**
     * Allocate the final image matrix.
     */
    void Calloc_image();

     /**
      * Clean the final image matrix.
      */
    void CClear();

    /**
    * Getters and setters of attributes for receiver object.
    */
    int getType() const;
    void setType(int type_);

    float3 getNormal() const;
    void setNormal(float3 normal);

    float3 getPosition() const;
    void setPosition(float3 pos);

    float3 getSize() const;
    void setSize(float3 size_);

    int getSurfaceIndex() const;
    void setSurfaceIndex(int surface_index);

    float getPixelLength() const;

    __host__ __device__ float *getDeviceImage() const;

    __host__ __device__ int2 getResolution() const;

protected:
    int type_;
    float3 normal_;
    float3 pos_;
    float3 size_;
    int surface_index_;        //the index of receiving surface
    float pixel_length_;
    float *d_image_;            //On GPU, memory size = resolution_.x * resolution_.y
    int2 resolution_;           //x means column, y means row
};


#endif //SOLARENERGY_CHIER_RECEIVER_CUH