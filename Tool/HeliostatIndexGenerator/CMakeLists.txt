message("----------Heliostat Index Generator cmake begins----------")

add_library(heliostat_index_generator STATIC GenerateHeliostatIndex.cpp GenerateHeliostatIndex.h)

message("----------Heliostat Index Generator cmake ends---------- ")