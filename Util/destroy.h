//
// Created by feng on 19-3-27.
// PS: free cpu or gpu memory.
//

#pragma once

#include <vector>

using namespace std;

namespace free_scene{
    template <typename T>
    inline void cpu_free(T *&t){
        if(t){
            delete t;
            t = nullptr;
        }
    }

    template <typename T>
    inline void gpu_free(T *&t){
        if(t){
            t->CClear();
        }
    }

    template <typename T>
    inline void cpu_free(vector<T *> &Tarray){
        for(int i = 0; i < Tarray.size(); ++i){
            cpu_free(Tarray[i]);
        }
    }

    template <typename T>
    inline void gpu_free(vector<T *> &Tarray){
        for(int i = 0; i < Tarray.size(); ++i){
            gpu_free(Tarray[i]);
        }
    }
}