#General Configuration
cmake_minimum_required(VERSION 3.13)
project(SolarEnergy_Chier)

#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

# packages
find_package(CUDA REQUIRED)

#Tool
add_subdirectory(Tool/RandomNumberGenerator)

#Data Structure
add_subdirectory(Scene)

#Set libs
#set(EXTRA_LIBS ${EXTRA_LIBS} rayTracingPipeline)

#Finial Project
cuda_add_executable(SolarEnergy_Chier main.cpp)
target_include_directories(SolarEnergy_Chier PUBLIC Util/)
target_include_directories(SolarEnergy_Chier PUBLIC Scene/)
target_include_directories(SolarEnergy_Chier PUBLIC Tool/)
target_link_libraries(SolarEnergy_Chier ${EXTRA_LIBS})

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")