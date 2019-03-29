# General Configuration
cmake_minimum_required(VERSION 3.12)
project(SolarEnergyRayTracing)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)

# Tool
add_subdirectory(Tool/RandomNumberGenerator)


# Data Structure
add_subdirectory(Scene)

# Quasi Monte Carlo Ray Tracing
add_subdirectory(QuasiMonteCarloRayTracing)

# Ray Tracing Pipeline
add_subdirectory(RayTracingPipeline)



# Set Libs
set(EXTRA_LIBS ${EXTRA_LIBS} rayTracingPipeline)

# Finial Project
cuda_add_executable(SolarEnergyRayTracing main.cpp)
target_link_libraries(SolarEnergyRayTracing ${EXTRA_LIBS})
target_include_directories(SolarEnergyRayTracing PUBLIC RayTracingPipeline/)

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")