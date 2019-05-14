//
// Created by feng on 19-4-17.
//
#include <chrono>

#include "RayTracingPipeline.h"
#include "ArgumentParser/ArgumentParser.h"
#include "SceneProcessor.h"
#include "SceneLoader.h"
#include "TaskLoader.h"
#include "global_function.cuh"
#include "QuasiMonteCarloRayTracer.h"
#include "ImageSaver/ImageSaver.h"
#include "Smoother/ImageSmoother.cuh"

void RayTracingPipeline::rayTracing(int argc, char **argv) {
    // 1. Pass argument
    //      1) solar scene file path
    //      2) configuration file path
    //      3) the file path saving the heliostats' indexes which will be ray-traced
    //      4) the output path
    std::cout << "1. Start loading arguments...";
    ArgumentParser *argumentParser = new ArgumentParser();
    argumentParser->parser(argc, argv);
    // 2. Initialize solar scene
    std::cout << "\n\n2. Initialize solar scene..." << std::endl;
    //  2.1 configuration
    std::cout << "\t2.1 Load configuration from '" << argumentParser->getConfigurationPath() << "'." << std::endl;
    SceneConfiguration *sceneConfiguration = SceneConfiguration::getInstance();
    sceneConfiguration->loadConfiguration(argumentParser->getConfigurationPath());

    //  2.2 load scene
    std::cout << "\t2.2 Load scene file from '" << argumentParser->getScenePath() << "'." << std::endl;
    SceneLoader sceneLoader;
    SolarScene *solarScene = SolarScene::GetInstance();
    sceneLoader.SceneFileRead(solarScene, argumentParser->getScenePath());

    //  2.3 process scene
    std::cout << "\t2.3 Process scene." << std::endl;
    SceneProcessor sceneProcessor(sceneConfiguration);

    sceneProcessor.processScene(solarScene);
    //  2.4 load heliostats indexes
    std::cout << "\t2.4 Load heliostats indexes from '" << argumentParser->getHeliostatIndexLoadPath() << "'."
              << std::endl;
    TaskLoader taskLoader;
    taskLoader.loadRayTracingHeliostatIndex(argumentParser->getHeliostatIndexLoadPath(), *solarScene);

    auto sumstart_time = std::chrono::high_resolution_clock::now();
    auto sumend_time = sumstart_time;
    long long sumelapsed;
    /**
     * Test 100 times here:
     */
    vector<float> max_values;
    vector<float> sum_values;

    for(int i = 0 ; i < 1; ++i){

        // 3. Ray tracing (could be parallel)
        std::cout << "\n3. Start ray tracing..." << std::endl;
        QuasiMonteCarloRayTracer QMCRTracer;

        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = start_time;
        long long elapsed;

        for (int heliostat_index : taskLoader.getHeliostatIndexesArray()) {
            try {
                // Count the time
                start_time = std::chrono::high_resolution_clock::now();

                QMCRTracer.rayTracing(solarScene, heliostat_index);

                end_time = std::chrono::high_resolution_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

                std::cout << "\tNo." << heliostat_index << " heliostats took " << elapsed << " microseconds to ray tracing." << std::endl;

            } catch (exception e) {
                std::cerr << "  Failure in No." << heliostat_index << " heliostat ray tracing." << std::endl;
            }
        }

        float max_value;
        // 4. Save results
        std::cout << "\n4. Save results in '" << argumentParser->getOutputPath() << "' directory." << std::endl;
        for (int receiver_index : taskLoader.getReceiverIndexesArray()) {
            std::cout << "------Saving No." << receiver_index << " receiver------" << std::endl;
            // Choose to save No.0 receiver max values.
            if(receiver_index == 0){
                max_value = saveReceiverResult(solarScene->getReceivers()[receiver_index],
                                               argumentParser->getOutputPath() + std::to_string(receiver_index),
                                               receiver_index);
            }else{
                saveReceiverResult(solarScene->getReceivers()[receiver_index],
                                   argumentParser->getOutputPath() + std::to_string(receiver_index),
                                   receiver_index);
            }

            solarScene->getReceivers()[receiver_index]->Cclean_image_content();
        }

        max_values.push_back(max_value);

    }


//    std::cout << "max_values====================" << std::endl;
//    for(float max_value : max_values){
//        std::cout << max_value << ",";
//    }
//    std::cout << std::endl;

    sumend_time = std::chrono::high_resolution_clock::now();
    sumelapsed = std::chrono::duration_cast<std::chrono::microseconds>(sumend_time - sumstart_time).count();

    std::cout << "All heliostats totally took " << sumelapsed << " microseconds to ray tracing." << std::endl;

    // 5. Clean up the scene
    solarScene->clear();
}


//Two smooth methods: trim mean & trim gaussian
float RayTracingPipeline::saveReceiverResult(Receiver *receiver, std::string pathAndName, int receiverIndex) {
    int2 resolution = receiver->getResolution();
    float *h_array = nullptr;
    float *d_array = receiver->getDeviceImage();    // Before smooth
    std::cout << "\treceiver resolution: (" << resolution.y << ", " << resolution.x << ").\n";
    global_func::gpu2cpu(h_array, d_array, resolution.x * resolution.y);
    float max_value = ImageSaver::saveText(pathAndName + "_receiver_before_smooth.txt" , resolution.y, resolution.x, h_array);

    auto start_time = std::chrono::high_resolution_clock::now();
//    //Image smooth method: trim mean smooth
//    ImageSmoother::image_smooth(d_array, 5, 0.05, resolution.x, resolution.y);          //TODO: TEST diff number here
//    std::cout << "【Trim Mean Smooth】" << std::endl;

    //Image smooth； trim gaussian smooth
    ImageSmoother::image_smooth(d_array, 5, 0.05, resolution.x, resolution.y, 1.0f);
    std::cout << "【Trim Gaussian Smooth】" << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "\t[Smooth time]:  " << elapsed << " microseconds." << std::endl;

    global_func::gpu2cpu(h_array, d_array, resolution.x * resolution.y);

    max_value = ImageSaver::saveText(pathAndName + "_receiver_after_smooth.txt", resolution.y, resolution.x, h_array);

    // clear
    delete(h_array);
    h_array = nullptr;
    d_array = nullptr;

    return max_value;
}