message("----------Image Saver cmake begins----------")

include_directories("${CMAKE_SOURCE_DIR}/Util")
add_library(image_saver STATIC ImageSaver.cpp ImageSaver.h)

message("----------Image Saver cmake ends---------- ")