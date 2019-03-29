message("QMCRT construction begins...")

# required cmake version
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB_RECURSE CURRENT_SOURCES  *.cpp *.cu)

cuda_add_library(quasiMonteCarloRayTracing STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_include_directories(datastructure PUBLIC "${CMAKE_SOURCE_DIR}/Util")
target_include_directories(datastructure PUBLIC "${CMAKE_SOURCE_DIR}/Scene")

target_link_libraries(quasiMonteCarloRayTracing datastructure)

message("QMCRT construction ends...")


message("CUDA_NVCC_FLAGS is ${CUDA_NVCC_FLAGS}")
message("CUDA_NVCC_FLAGS_RELEASE is ${CUDA_NVCC_FLAGS_RELEASE}")
message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")