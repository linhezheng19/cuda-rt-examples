#ifndef SAMPLE_H
#define SAMPLE_H

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "NvInfer.h"

using namespace std;
using namespace nvinfer1;

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            cerr << "CUDA error: " << error_code << "! at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(0);                                                                         \
        }                                                                                      \
    }
#endif

class EngineLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override
    {
        if (severity != Severity::kVERBOSE)
            std::cout << msg << std::endl;
    }
};

class SampleAlexnet {
public:
    SampleAlexnet(int batch_size, int input_h, int input_w, int num_classes);
    ~SampleAlexnet();

    void build(const string& weight_file, const string& engine_file);
    void doInference(float* input, float* output);

private:
    map<string, nvinfer1::Weights> loadWeights(const string& weight_file);
    bool createEngine(IBuilder* builder, DataType type, const string& weight_file);
    bool serializeEngine(const string& engine_file);
    bool deserializeEngine(const string& engine_file);

private:
    int BATCH_SIZE = 1;
    int INPUT_H = 224;
    int INPUT_W = 224;
    int NUM_CLASSES = 1000;
    char* INPUT_BLOB_NAME = "input";
    char* OUTPUT_BLOB_NAME = "output";
    EngineLogger gLogger;
    ICudaEngine* mEngine;
    IExecutionContext* mContext;
};

#endif  // SAMPLE_H
