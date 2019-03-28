#General Configuration
cmake_minimum_required(VERSION 3.13)
project(SolarEnergy_Chier)

#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)

#Data Structure
add_subdirectory(Scene)

#Tool
add_subdirectory(Tool/RandomNumberGenerator)

#Set libs
set(EXTRA_LIBS ${EXTRA_LIBS} rayTracingPipeline)

#Finial Project
cuda_add_executable(SolarEnergy_Chier main.cpp)
target_link_libraries(SolarEnergy_Chier ${EXTRA_LIBS})
target_include_directories(SolarEnergy_Chier PUBLIC Util/)


file(GLOB_RECURSE CURRENT_HEADERS *.h *.hpp *.cuh)
file(GLOB_RECURSE CURRENT_SOURCES *.cpp *.cu)
source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})


message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")