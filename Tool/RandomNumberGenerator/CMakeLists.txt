message("Random Generator cmake begins...")

include_directories("${CMAKE_SOURCE_DIR}/Util")

#packages
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -O3;-G;-g)

cuda_add_library(randomGenerator STATIC RandomGenerator.cuh RandomGenerator.cpp RandomGenerator.cu)
target_link_libraries(randomGenerator ${CUDA_curand_LIBRARY})

message("Random Generator cmake ends...")