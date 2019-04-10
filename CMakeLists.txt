# General Configuration
cmake_minimum_required(VERSION 3.12)
project(SolarEnergy_Chier)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)

# Tool
add_subdirectory(Tool/RandomNumberGenerator)
add_subdirectory(Tool/FileLoader)
add_subdirectory(Tool/ArgumentParser)

# Data Structure
add_subdirectory(Scene)

# Quasi Monte Carlo Ray Tracing
add_subdirectory(QuasiMonteCarloRayTracing)

# Ray Tracing Pipeline
add_subdirectory(RayTracingPipeline)

#Tests
set(GOOGLE_TEST_PATH Tests)
add_subdirectory(${GOOGLE_TEST_PATH})

# Set Libs
set(EXTRA_LIBS ${EXTRA_LIBS} rayTracingPipeline)

# Finial Project
cuda_add_executable(SolarEnergy_Chier main.cpp)
target_link_libraries(SolarEnergy_Chier ${EXTRA_LIBS})
target_include_directories(SolarEnergy_Chier PUBLIC RayTracingPipeline/)

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")