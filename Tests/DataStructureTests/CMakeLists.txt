message("----------Data structure test cmake begins----------")

cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# package
find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
#set(CUDA_NVCC_FLAGS -O3;-G;-g)

# Test target
file(GLOB CURRENT_SOURCES *.cpp *.cu)
message("Test files name: ${CURRENT_SOURCES}")
cuda_add_executable(DataStructureTests ${CURRENT_SOURCES})
target_link_libraries(DataStructureTests gtest gtest_main)
target_link_libraries(DataStructureTests gmock gmock_main)

message("chier-test: ${gmock_SOURCE_DIR}")
message("chier-test: ${gtest_SOURCE_DIR}")

# Add the files to be tested
include_directories("${CMAKE_SOURCE_DIR}/Scene")
target_link_libraries(DataStructureTests datastructure)

message("----------Data structure test construction ends----------")