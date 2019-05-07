//
// Created by feng on 19-4-22.
//

#ifndef SOLARENERGY_CHIER_HEAP_CUH
#define SOLARENERGY_CHIER_HEAP_CUH

#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

/**
 *  class: Heap
 */
class Heap {
//private:
public:
    int numElems;
    int max_size;
    bool (*cmp_func)(const float &, const float &);

    float *elements;

    float2 *elements_2;
    __device__ __host__ Heap(int size, float *element_entry) : numElems(1), max_size(size + 1), elements(element_entry){}
    __device__ __host__ Heap(int size, float2 *element_entry) : numElems(1), max_size(size + 1), elements_2(element_entry){}

    __device__ __host__ bool insert(float elem){
        if(full()){
            return false;
        }
        int i = numElems++;
        for(; i != 1 && cmp_func(elem, elements[i/2]); i /= 2){
            elements[i] = elements[i/2];
        }
        elements[i] = elem;
        return true;
    }

    __device__ __host__ bool insert2(float2 elem){
        if(full()){
            return false;
        }
        int i = numElems++;
        for(; i != 1 && cmp_func(elem.x, elements_2[i/2].x); i /= 2){
            elements_2[i] = elements_2[i/2];
        }
        elements_2[i] = elem;
        return true;
    }


    __device__ __host__ float delete_t(){
        if(empty()){
            return -100;
        }
        float return_value = elements[1];
        float last_value = elements[--numElems];
        int i = 1, cmp_id;
        for(; 2 * i < numElems;){
            if(2 * i + 1 >= numElems ||                             // only has left child
                cmp_func(elements[2 * i], elements[2 * i + 1])){    // have 2 children
                cmp_id = 2 * i;
            }else{
                cmp_id = 2 * i  + 1;
            }

            if(cmp_func(elements[cmp_id], last_value)){
                elements[i] = elements[cmp_id];
                i = cmp_id;
            }else{
                break;
            }
        }
        elements[i] = last_value;
        return return_value;
    }

    __device__ __host__ float2 delete_t2(){
        if(empty()){
            return make_float2(-100,-100);
        }
        float2 return_value = elements_2[1];
        float2 last_value = elements_2[--numElems];
        int i = 1, cmp_id;
        for(; 2 * i < numElems;){
            if(2 * i + 1 >= numElems ||                             // only has left child
               cmp_func(elements_2[2 * i].x, elements_2[2 * i + 1].x)){    // have 2 children
                cmp_id = 2 * i;
            }else{
                cmp_id = 2 * i  + 1;
            }

            if(cmp_func(elements_2[cmp_id].x, last_value.x)){
                elements_2[i].x = elements_2[cmp_id].x;
                i = cmp_id;
            }else{
                break;
            }
        }
        elements_2[i] = last_value;
        return return_value;
    }


    __device__ __host__ float delete_and_insert(float elem){
        if(empty()){
            return -100;
        }
        float return_value = elements[1];
        int i = 1, cmp_id;
        for(; 2 * i < numElems;){
            // only has left child
            cmp_id = (2 * i + 1 >= numElems || cmp_func(elements[2 * i], elements[2 * i + 1])) ? 2 * i : 2 * i + 1;

            if(cmp_func(elements[cmp_id], elem)){
                elements[i] = elements[cmp_id];
                i = cmp_id;
            }else{
                break;
            }
        }
        elements[i] = elem;
        return return_value;
    }


    __device__ __host__ float2 delete_and_insert2(float2 elem){
        if(empty()){
            return make_float2(-100,-100);
        }
        float2 return_value = elements_2[1];
        int i = 1, cmp_id;
        for(; 2 * i < numElems;){
            // only has left child
            cmp_id = (2 * i + 1 >= numElems || cmp_func(elements_2[2 * i].x, elements_2[2 * i + 1].x)) ? 2 * i : 2 * i + 1;

            if(cmp_func(elements_2[cmp_id].x, elem.x)){
                elements_2[i] = elements_2[cmp_id];
                i = cmp_id;
            }else{
                break;
            }
        }
        elements_2[i] = elem;
        return return_value;
    }


    __device__ __host__ float top(){
        if(!empty()){
            return elements[1];
        }
        printf("Error------------no top member!!!!!\n");
        return 0;
    }

    __device__ __host__ float2 top2(){
        if(!empty()){
            return elements_2[1];
        }
        printf("Error------------no top member!!!!!\n");
        return make_float2(0,0);
    }

    __device__ __host__ int size() const{
        return numElems;
    }

    __device__ __host__ bool full() const{
        return numElems == max_size;
    }

    __device__ __host__ bool empty() const{
        return numElems == 1;
    }

    __device__ __host__ float sum() const{
        float sum = 0.0f;
        for(int i = 1; i < numElems; ++i){
            sum += elements[i];
        }
        return sum;
    }

    __device__ __host__ float sum2() const{
        float sum = 0.0f;
        for(int i = 1; i < numElems; ++i){
            sum += elements_2[i].x * elements_2[i].y;
        }
        return sum;
    }

    __device__ __host__ float sumy() const{
        float sum = 0.0f;
        for(int i = 1; i < numElems; ++i){
            sum += elements_2[i].y;
        }
        return sum;
    }


    __device__ __host__ void set_cmp_func(bool(*compare)(const float &, const float &)){
        cmp_func = compare;
    }

    friend class Max_heap;
    friend class Min_heap;
};



/**
 *  class: Max_heap
 */
inline __device__ __host__ bool larger(const float &t1, const float &t2){
    return t1 > t2;
}


class Max_heap {
private:
    Heap heap;

public:
    __device__ __host__ Max_heap(int size, float *element_entry) : heap(size, element_entry){
        heap.set_cmp_func(larger);
    }

    __device__ __host__ Max_heap(int size, float2 *element_entry) : heap(size, element_entry){
        heap.set_cmp_func(larger);
    }


    __device__ __host__ bool insert(float elem){
        return heap.insert(elem);
    }

    __device__ __host__ bool insert2(float2 elem){
        return heap.insert2(elem);
    }

    __device__ __host__ float delete_t(){
        return heap.delete_t();
    }

    __device__ __host__ float2 delete_t2(){
        return heap.delete_t2();
    }

    __device__ __host__ float delete_and_insert(float elem){
        return heap.delete_and_insert(elem);
    }

    __device__ __host__ float2 delete_and_insert2(float2 elem){
        return heap.delete_and_insert2(elem);
    }

    __device__ __host__ float top(){
        return heap.top();
    }

    __device__ __host__ float2 top2(){
        return heap.top2();
    }


    __device__ __host__ int size() const{
        return heap.size();
    }

    __device__ __host__ bool empty() const{
        return heap.empty();
    }

    __device__ __host__ bool full() const{
        return heap.full();
    }

    __device__ __host__ float sum() const{
        return heap.sum();
    }

    __device__ __host__ float sum2() const{
        return heap.sum2();
    }

    __device__ __host__ float sumy() const{
        return heap.sumy();
    }
};



/**
 *  class: Min_heap
 */

inline __device__ __host__ bool smaller(const float &t1, const float &t2){
    return t1 < t2;
}


class Min_heap {
private:
    Heap heap;

public:
    __device__ __host__ Min_heap(int size, float *element_entry) : heap(size, element_entry){
        heap.set_cmp_func(smaller);
    }

    __device__ __host__ Min_heap(int size, float2 *element_entry) : heap(size, element_entry){
        heap.set_cmp_func(smaller);
    }

    __device__ __host__ bool insert(float elem){
        return heap.insert(elem);
    }

    __device__ __host__ bool insert2(float2 elem){
        return heap.insert2(elem);
    }

    __device__ __host__ float delete_t(){
        return heap.delete_t();
    }

    __device__ __host__ float2 delete_t2(){
        return heap.delete_t2();
    }

    __device__ __host__ float delete_and_insert(float elem){
        return heap.delete_and_insert(elem);
    }

    __device__ __host__ float2 delete_and_insert2(float2 elem){
        return heap.delete_and_insert2(elem);
    }

    __device__ __host__ float top(){
        return heap.top();
    }

    __device__ __host__ float2 top2(){
        return heap.top2();
    }


    __device__ __host__ int size() const{
        return heap.size();
    }

    __device__ __host__ bool empty() const{
        return heap.empty();
    }

    __device__ __host__ bool full() const{
        return heap.full();
    }

    __device__ __host__ float sum() const{
        return heap.sum();
    }

    __device__ __host__ float sum2() const{
        return heap.sum2();
    }

    __device__ __host__ float sumy() const{
        return heap.sumy();
    }

};


#endif //SOLARENERGY_CHIER_HEAP_CUH