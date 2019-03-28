message("Scene construction begins...")

#required cmake version
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

#packages
find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -00")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -00")

file(GLOB_RECURSE CURRENT_HEADERS *.h *.hpp *.cuh)
#list(FILTER CURRENT_HEADERS EXCLUDE REGEX "json-develop/.*") #exclude files under json-develop/
file(GLOB_RECURSE CURRENT_SOURCES *.cpp *.cu)
#list(FILTER CURRENT_SOURCES EXCLUDE REGEX "json-develop/.*") #exclude files under json-develop/

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

#Add files under DataStructure/ and other tools like Tool or Util
cuda_add_library(datastructure STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_include_directories(datastructure PUBLIC "${CMAKE_SOURCE_DIR}/Tool")
target_include_directories(datastructure PUBLIC "${CMAKE_SOURCE_DIR}/Util")
target_include_directories(datastructure PUBLIC "DataStructure/Grid")
target_include_directories(datastructure PUBLIC "DataStructure/Sunray")
target_include_directories(datastructure PUBLIC "DataStructure/Heliostat")
target_include_directories(datastructure PUBLIC "DataStructure/Receiver")

target_link_libraries(datastructure randomGenerator)

message("Scene construction ends...")

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
