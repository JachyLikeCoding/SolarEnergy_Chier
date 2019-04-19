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

    sceneProcessor.processScene(solarScene);    //-------------------------has bug
    //  2.4 load heliostats indexes
    std::cout << "\t2.4 Load heliostats indexes from '" << argumentParser->getHeliostatIndexLoadPath() << "'."
              << std::endl;
    TaskLoader taskLoader;
    taskLoader.loadRayTracingHeliostatIndex(argumentParser->getHeliostatIndexLoadPath(), *solarScene);

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

            std::cout << "\tNo." << heliostat_index << " heliostats took " << elapsed << " microseconds." << std::endl;

        } catch (exception e) {
            std::cerr << "  Failure in No." << heliostat_index << " heliostat ray tracing." << std::endl;
        }
    }

    // 4. Save results
    std::cout << "\n4. Save results in '" << argumentParser->getOutputPath() << "' directory." << std::endl;
    for (int receiver_index : taskLoader.getReceiverIndexesArray()) {
        std::cout << "  Saving No." << receiver_index << " receiver." << std::endl;
        saveReceiverResult(solarScene->getReceivers()[receiver_index],
                           argumentParser->getOutputPath() + std::to_string(receiver_index) + "_receiver.txt");
    }

    // 5. Clean up the scene
    solarScene->clear();
}


void RayTracingPipeline::saveReceiverResult(Receiver *receiver, std::string pathAndName) {
    int2 resolution = receiver->getResolution();
    float *h_array = nullptr;
    float *d_array = receiver->getDeviceImage();
    std::cout << " resolution: (" << resolution.y << ", " << resolution.x << ").";
    global_func::gpu2cpu(h_array, d_array, resolution.x * resolution.y);
    ImageSaver::saveText(pathAndName, resolution.y, resolution.x, h_array);

    // clear
    delete(h_array);
    h_array = nullptr;
    d_array = nullptr;
}