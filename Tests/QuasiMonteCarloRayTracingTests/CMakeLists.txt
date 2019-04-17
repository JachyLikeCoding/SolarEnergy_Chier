message("----------QMCRT Tests cmake begins----------")

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")


# Test targets
file(GLOB CURRENT_SOURCES *.cpp *.cu)
message("Test files name: ${CURRENT_SOURCES}")
cuda_add_executable(QMCRT_Tests ${CURRENT_SOURCES})
target_link_libraries(QMCRT_Tests gtest gtest_main)
target_link_libraries(QMCRT_Tests gmock gmock_main)

message("chier-test: ${gmock_SOURCE_DIR}")
message("chier-test: ${gtest_SOURCE_DIR}")

# Add the files to be tested
include_directories("${CMAKE_SOURCE_DIR}/QuasiMonteCarloRayTracing")
target_link_libraries(QMCRT_Tests quasiMonteCarloRayTracing)

message("----------QMCRT Tests cmake ends----------")
