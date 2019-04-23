message("----------Image Smoother cmake begins----------")

# packages
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS -O3;-G;-g)

include_directories("${CMAKE_SOURCE_DIR}/Util")

cuda_add_library(image_smoother STATIC ImageSmoother.cuh ImageSmoother.cu heap.cuh)
target_link_libraries(image_smoother ${CUDA_curand_LIBRARY})

message("----------Image Smoother cmake ends---------- ")