message("----------Ray Tracing Pipeline begins----------")

# General Configuration
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# Generate lib
file(GLOB_RECURSE CURRENT_HEADERS  *.h)
file(GLOB_RECURSE CURRENT_SOURCES  *.cpp)
cuda_add_library(rayTracingPipeline STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

# Add datastructure lib
target_include_directories(rayTracingPipeline PUBLIC "${CMAKE_SOURCE_DIR}/Util")
target_include_directories(rayTracingPipeline PUBLIC "${CMAKE_SOURCE_DIR}/Scene")
include_directories("${CMAKE_SOURCE_DIR}/QuasiMonteCarloRayTracing")
include_directories("${CMAKE_SOURCE_DIR}/Tool")


# Set Libs
set(EXTRA_LIBS ${EXTRA_LIBS} argumentParser image_saver image_smoother quasiMonteCarloRayTracing datastructure)
target_link_libraries(rayTracingPipeline ${EXTRA_LIBS})

message("----------Ray Tracing Pipeline ends----------")