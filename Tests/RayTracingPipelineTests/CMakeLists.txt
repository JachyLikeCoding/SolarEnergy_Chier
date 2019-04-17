message("----------Ray Tracing Pipeline Tests cmake begins ---------- ")

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# Test Target
file(GLOB CURRENT_SOURCES *.cpp)
message("Test files name: ${CURRENT_SOURCES}")
cuda_add_executable(RayTracingPipelineTests ${CURRENT_SOURCES})
target_link_libraries(RayTracingPipelineTests gtest gtest_main)

# Add Google Tests
include_directories("${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}")

# Add the files to be tested
include_directories("${CMAKE_SOURCE_DIR}/RayTracingPipeline")
target_link_libraries(RayTracingPipelineTests rayTracingPipeline)

message("----------Ray Tracing Pipeline Tests Test cmake ends ---------- ")