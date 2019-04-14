#include <math.h>
#include "SceneProcessor.h"
#include "global_function.cuh"
#include "RandomNumberGenerator/RandomGenerator.cuh"

namespace samplelights{
    float sunshape_intensity(float theta, const float &k, const float &gamma){
        //theta must be in range [0 , 4.65]
        return cosf(0.326f * theta) / cosf(0.308f * theta);
    }

    float integration_larger_than_465_intensity(float theta, float k, float gamma){
        //theta must be in range (4.65 , 9.3]
        return expf(k) / (gamma + 1) * powf(theta, gamma + 1);
    }

    //Description:
    //  - h_k:(return value) slope (length_interval / subarea)
    //  - num_group: num of intervals
    //  - k, gamma: used in intensity calculation
    //  -upper_lm: length_interval = upper_lm / num_group

    float2 *parameters_generate(float *h_k, int num_group, float k, float gamma, float upper_lm){
        float length_interval = upper_lm / float(num_group);
        float x = 0.0f;
        float2 *h_cdf = new float2[num_group + 1];
        h_cdf[0].x = 0.0f;
        h_cdf[0].y = 0.0f;
        float hist_pre = sunshape_intensity(x, k, gamma);
        float hist_current;
        float subarea;
        for(int i = 1; i <= num_group; ++i){
            x += length_interval;
            hist_current = sunshape_intensity(x, k, gamma);
            subarea = (hist_current + hist_pre) / 2.0f * length_interval;

            h_cdf[i].x = x;
            h_cdf[i].y = subarea + h_cdf[i-1].y;
            h_k[i-1] = length_interval / subarea;

            hist_pre = hist_current;
        }
        return h_cdf;
    }

    __host__ __device__ int max_less_index(float2 *d_cdf, float value, size_t n){   // d_cdf[n+1]
        int left = 0, right = n;
        int mid;
        while(left <= right){
            mid = (left + right) >> 1;
            if(value > d_cdf[mid].y){
                left = mid + 1;
            }else if(value < d_cdf[mid].y){
                right = mid - 1;
            }else
                return mid;
        }
        return right;
    }

    __global__ void linear_interpolation(float *d_0_1, float2 *d_cdf, float *d_k, float integration_less_than_465,
                                            float gamma, float A, float B, float C, size_t n, size_t size){
        const int myId = global_func::getThreadID();
        if(myId >= size)
            return;

        float u = d_0_1[myId] * A;
        if(u < integration_less_than_465){
            int id = max_less_index(d_cdf, u, n);
            u = (u - d_cdf[id].y) * d_k[id] + d_cdf[id].x;
            u = atan(sqrtf(u/4.65f) * 0.5016916f);  // tan(0.465rad) = 0.5016916f
        }else{
            u = powf((u - integration_less_than_465) * B + C, 1 / (gamma+1));

            float ri2 = 0.5016916f * 0.5016916f;    // tan(0.465rad) = 0.5016916f
            float ro2 = 1.340874f * 1.340874f;      // tan(0.93rad) = 1.340874f
            u = atan(sqrt(ro2 + (u - 4.65f) / 4.65f * (ro2 - ri2)));
        }
        d_0_1[myId] = u / 100.0f;
        return;
    }

    __global__ void map_permuation(float3 *d_turbulance, const float *d_x, const float *d_z, const size_t size){
        unsigned int myId = global_func::getThreadID();
        if(myId >= size)
            return;

        d_turbulance[myId].x = d_x[myId];
        d_turbulance[myId].y = 1.0f;
        d_turbulance[myId].z = d_z[myId];

        d_turbulance[myId] = normalize(d_turbulance[myId]);
    }

    __global__ void map_angle2xyz(float3 *d_turbulance, const float *d_nonUniform, const float *d_uniform, const size_t size){
        unsigned int myId = global_func::getThreadID();
        if(myId >= size)
            return;

        float theta = d_nonUniform[myId];
        float phi = d_uniform[myId] * 2 * MATH_PI;
        d_turbulance[myId] = global_func::angle2xyz(make_float2(theta, phi));
    }

    /**
     * For SceneProcessorTest.cpp
     */
     void draw_distribution(float *d_float_array, int size, float lower_bound, float upper_bound, std::string random_number_name){
         const int nstars = 1000;   //maximum number of stars to distribute
         const int nHist = 20;

         vector<int> histogram(nHist, 0);

         float *h_float_array = nullptr;
         global_func::gpu2cpu(h_float_array, d_float_array, size);

         float gap = (upper_bound - lower_bound) / float(nHist);

         for(int i = 0; i < size; ++i){
             if(h_float_array[i] > lower_bound && h_float_array[i] < upper_bound){
                 int n = int((h_float_array[i] - lower_bound) / gap);
                 ++histogram[n];
             }
         }

         std::cout << "Distribution of " << random_number_name << ":" << std::endl;
         std::cout << std::setprecision(2) << lower_bound << endl;

         for(int h : histogram){
             std::cout << "\t" << std::string(h * nstars / size, '*') << std::endl;
         }
         std::cout << std::setprecision(2) << upper_bound << endl;

         delete[] h_float_array;
         h_float_array = nullptr;
     }

}


/**
 * Permutation: Generate sample lights with
 *  - theta ~ G(0, disturb_std)
 *  - phi ~ Uniform(0, 2pi)
 */

bool SceneProcessor::set_perturbation(Sunray &sunray) {
    if(sunray.getDevicePerturbation()){
        //Make sure the sunray perturbation is empty.
        //If not , clean the device perturbation before calling this method.
        return false;
    }

    int size = sunray.getNumOfSunshapeGroups() * sunray.getNumOfSunshapeLightsPerGroup();

    //Step 1: Allocate memory for sunray.d_permutation_ on GPU
    float3 *d_permutation = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_permutation, sizeof(float3) * size));

    //Step 2: Allocate memory for theta and phi
    float *d_gaussian_x = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_gaussian_x, sizeof(float) * size));
    float *d_gaussian_z = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_gaussian_z, sizeof(float) * size));

    //Step 3: Generate theta and phi
    RandomGenerator::gpu_Gaussian(d_gaussian_x, 0.0f, sceneConfiguration->getDisturb_std(), size);
    RandomGenerator::gpu_Gaussian(d_gaussian_z, 0.0f, sceneConfiguration->getDisturb_std(), size);

//#ifdef SCENE_PROCESSOR_TEST_CPP
    samplelights::draw_distribution(d_gaussian_x, size, -0.050f, 0.050f, "theta");
    samplelights::draw_distribution(d_gaussian_z, size, -0.050f, 0.050f, "phi");
//#endif

    //Step 4: (theta, phi) -> (x, y, z)
    int nThreads;
    dim3 nBlocks;
    global_func::setThreadBlocks(nBlocks, nThreads, size);
    samplelights::map_permuation << < nBlocks, nThreads >> > (d_permutation, d_gaussian_x, d_gaussian_z, size);
    sunray.setDevicePerturbation(d_permutation);

    // Step 5: Clean up
    checkCudaErrors(cudaFree(d_gaussian_x));
    checkCudaErrors(cudaFree(d_gaussian_z));

    return true;
}


/**
 * sampleLights: Generate sample lights with
 *  - theta ~ Buie distribution
 *  - phi ~ Uniform(0, 2pi)
 */

bool SceneProcessor::set_samplelights(Sunray &sunray) {
    if(sunray.getDeviceSampleLights()){
        //Make sure the sunray.samplelights are empty.
        //If not, clean the device sample lights befire calling this method.
        return false;
    }

    // Input parameters
    int num_group = sceneConfiguration->getInverse_transform_sampling_groups();
    float csr = sunray.getCSR();
    float upper_lm = 4.65f;
    //κ = 0.9 ln(13.5χ) χ −0.3
    //γ = 2.2 ln(0.52χ) χ 0.43 − 0.1
    float k = 0.9f * logf(13.5f * csr) * powf(csr, -0.3f);
    float gamma = 2.2f * logf(0.52f * csr) * powf(csr, 0.43f) - 0.1f;
    float integration_value_between_465_930 = samplelights::integration_larger_than_465_intensity(9.3f, k, gamma) -
                                                samplelights::integration_larger_than_465_intensity(upper_lm, k, gamma);
    float *h_k = new float[num_group];
    float2 *h_cdf = samplelights::parameters_generate(h_k, num_group, k, gamma, upper_lm);
    float value_less_465 = h_cdf[num_group].y;

    float2 *d_cdf = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_cdf, sizeof(float2) * (num_group + 1)));
    checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, sizeof(float2) * (num_group + 1), cudaMemcpyHostToDevice));

    float *d_k = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_k, sizeof(float) * num_group));
    checkCudaErrors(cudaMemcpy(d_k, h_k, sizeof(float) * num_group, cudaMemcpyHostToDevice));

    // Generate uniform random theta and phi in range [0,1]
    float *d_theta = nullptr;
    int num_random = sunray.getNumOfSunshapeLightsPerGroup() * sunray.getNumOfSunshapeGroups();
    checkCudaErrors(cudaMalloc((void **) &d_theta, sizeof(float) * num_random));
    RandomGenerator::gpu_Uniform(d_theta, num_random);

    float *d_phi = nullptr;
    checkCudaErrors(cudaMalloc((void **) &d_phi, sizeof(float) * num_random));
    RandomGenerator::gpu_Uniform(d_phi, num_random);

    int nThreads = 1024;
    int nBlocks = (num_random + nThreads - 1) / nThreads;
    float A = value_less_465 + integration_value_between_465_930;
    float B = (gamma + 1) / expf(k);
    float C = powf(upper_lm, gamma + 1);

    // Change to correct theta
    samplelights::linear_interpolation << < nBlocks, nThreads >> >
            (d_theta, d_cdf, d_k, value_less_465, gamma, A, B, C, num_group, num_random);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

//#ifdef SCENE_PROCESSOR_TEST_CPP
    samplelights::draw_distribution(d_theta, num_random, 0.0f, 0.0093f, "theta");
    samplelights::draw_distribution(d_phi, num_random, 0.0f, 1.0f, "phi");
//#endif
    float3 *d_samplelights = nullptr;
    cudaMalloc((void **) &d_samplelights, sizeof(float3) * num_random);
    samplelights::map_angle2xyz << < nBlocks, nThreads >> > (d_samplelights, d_theta, d_phi, num_random);
    sunray.setDeviceSampleLights(d_samplelights);

    // Clean up
    delete[] h_k;
    delete[] h_cdf;
    h_k = nullptr;
    d_cdf = nullptr;
    checkCudaErrors(cudaFree(d_phi));
    checkCudaErrors(cudaFree(d_theta));
    checkCudaErrors(cudaFree(d_cdf));
    checkCudaErrors(cudaFree(d_k));
    d_phi = nullptr;
    d_theta = nullptr;
    d_cdf = nullptr;
    d_k = nullptr;

    return true;
}


















