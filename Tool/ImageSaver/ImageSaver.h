//
// Created by feng on 19-4-17.
//

#ifndef SOLARENERGY_CHIER_IMAGESAVER_H
#define SOLARENERGY_CHIER_IMAGESAVER_H

#include <string>

class ImageSaver{
public:
    static void saveText(std::string filename, int height, int width, float *h_data, int precision = 2, int rows_package = 10);
};



#endif //SOLARENERGY_CHIER_IMAGESAVER_H
