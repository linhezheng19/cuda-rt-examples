#include "sample.h"

#include <fstream>
#include "cuda_runtime_api.h"

SampleAlexnet::SampleAlexnet(int batch_size, int input_h, int input_w, int num_classes)
        : BATCH_SIZE(batch_size), INPUT_H(input_h), INPUT_W(input_w), NUM_CLASSES(num_classes) {
    mEngine = nullptr;
    mContext = nullptr;
}

SampleAlexnet::~SampleAlexnet() {
//    mEngine->destroy();
    mContext->destroy();
}

bool SampleAlexnet::serializeEngine(const string& engine_file) {
    IHostMemory* stream{nullptr};
    stream = mEngine->serialize();

    std::ofstream p(engine_file);
    if (!p)
    {
        cerr << "could not open plan output file" << endl;
        return false;
    }
    p.write(reinterpret_cast<const char*>(stream->data()), stream->size());
    stream->destroy();
    return true;
}

bool SampleAlexnet::deserializeEngine(const string& engine_file) {
    char* engine_stream;
    size_t size;
    std::ifstream file(engine_file, std::ios::binary);
    if (file.is_open()) {
        cout << "deserialize Engine!" << endl;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        engine_stream = new char[size];
        assert(engine_stream);
        file.read(engine_stream, size);
        file.close();
    } else {
        cerr << "Engine file is not exist!" << endl;
        return false;
    }
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    mEngine = runtime->deserializeCudaEngine(engine_stream, size, nullptr);
    runtime->destroy();
    return true;
}

map<string, nvinfer1::Weights> SampleAlexnet::loadWeights(const string& weight_file) {
    cout << "Loading weights ... " << endl;
    map<string, nvinfer1::Weights> weights;
    // open weights file
    ifstream input(weight_file);
    assert(input.is_open() && "Open weight file failed!");
    // start read weights
    int32_t num_layers;
    input >> num_layers;
    assert(num_layers > 0 && "Weight is wrong!");

    while (num_layers--) {
        // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_weights.html
        nvinfer1::Weights w{nvinfer1::DataType::kFLOAT, nullptr, 0};
        // read name and count
        string layer_name;
        uint32_t size;
        input >> layer_name >> std::dec >> size;
        // read weight's value
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t i = 0; i < size; ++i) {
            input >> std::hex >> val[i];
        }
        w.values = val;
        w.count = size;
        weights[layer_name] = w;
    }
    return weights;
}

bool SampleAlexnet::createEngine(IBuilder* builder, DataType type, const string& weight_file) {
    INetworkDefinition* net = builder->createNetworkV2(0U);  // fix input chw, not bchw
    IBuilderConfig* config = builder->createBuilderConfig();
    // define input
    ITensor* input = net->addInput(INPUT_BLOB_NAME, type, nvinfer1::Dims3{3, INPUT_H, INPUT_W});
    assert(input && "create input failed!");

    // load weights from file
    auto weights = loadWeights(weight_file);
    // define network
    // load weights according to the layer name in pytorch model
    IConvolutionLayer* conv1 = net->addConvolutionNd(*input, 64, DimsHW{11, 11}, weights["features.0.weight"], weights["features.0.bias"]);
    assert(conv1 && "create conv1 failed!");
    conv1->setStrideNd(DimsHW{4, 4});
    conv1->setPaddingNd(DimsHW{2, 2});
    // add activation
    IActivationLayer* relu1 = net->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(conv1 && "create relu1 failed!");
    // add pooling layer
    IPoolingLayer* pool1 = net->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW(3, 3));
    assert(conv1 && "create pool1 failed!");
    pool1->setStrideNd(DimsHW{2, 2});

    IConvolutionLayer* conv2 = net->addConvolutionNd(*pool1->getOutput(0),  192, DimsHW{5, 5}, weights["features.3.weight"], weights["features.3.bias"]);
    assert(conv1 && "create conv2 failed!");
    conv2->setPaddingNd(DimsHW{2, 2});
    IActivationLayer* relu2 = net->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(conv1 && "create relu2 failed!");
    IPoolingLayer* pool2 = net->addPooling(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(conv1 && "create pool2 failed!");
    pool2->setStrideNd(DimsHW{2, 2});

    IConvolutionLayer* conv3 = net->addConvolutionNd(*pool2->getOutput(0), 384, DimsHW{3, 3}, weights["features.6.weight"], weights["features.6.bias"]);
    assert(conv1 && "create conv3 failed!");
    conv3->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu3 = net->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    assert(conv1 && "create relu3 failed!");

    IConvolutionLayer* conv4 = net->addConvolutionNd(*relu3->getOutput(0), 256, DimsHW{3, 3}, weights["features.8.weight"], weights["features.8.bias"]);
    assert(conv4 && "conv4 create failed!");
    conv4->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu4 = net->addActivation(*conv4->getOutput(0), ActivationType::kRELU);
    assert(relu4 && "relu4 create failed!");

    IConvolutionLayer* conv5 = net->addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3, 3}, weights["features.10.weight"], weights["features.10.bias"]);
    assert(conv5  && "conv5 create failed!");
    conv5->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu5 = net->addActivation(*conv5->getOutput(0), ActivationType::kRELU);
    assert(relu5  && "relu5 create failed!");
    IPoolingLayer* pool3 = net->addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool3  && "pool3 create failed!");
    pool3->setStrideNd(DimsHW{2, 2});

    IFullyConnectedLayer* fc1 = net->addFullyConnected(*pool3->getOutput(0), 4096, weights["classifier.1.weight"], weights["classifier.1.bias"]);
    assert(conv1 && "create fc1 failed!");
    IActivationLayer* relu6 = net->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(conv1 && "create relu6 failed!");

    IFullyConnectedLayer* fc2 = net->addFullyConnected(*relu6->getOutput(0), 4096, weights["classifier.4.weight"], weights["classifier.4.bias"]);
    assert(conv1 && "create fc2 failed!");
    IActivationLayer* relu7 = net->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
    assert(conv1 && "create relu7 failed!");

    IFullyConnectedLayer* fc3 = net->addFullyConnected(*relu7->getOutput(0), 1000, weights["classifier.6.weight"], weights["classifier.6.bias"]);
    assert(conv1 && "create fc3 failed!");

    // mark output
    fc3->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    net->markOutput(*fc3->getOutput(0));

    // build engine
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(1 << 20);
    mEngine = builder->buildEngineWithConfig(*net, *config);

    // release resource
    for (auto& w : weights)
    {
        free((void*) (w.second.values));
    }
    net->destroy();
    return true;
}

void SampleAlexnet::build(const string& weight_file, const string& engine_file) {
    // build network
//    IBuilder* builder = createInferBuilder(gLogger);
//    createEngine(builder, DataType::kFLOAT, weight_file);
    if (!deserializeEngine(engine_file)) {
        IBuilder* builder = createInferBuilder(gLogger);
        createEngine(builder, DataType::kFLOAT, weight_file);
        serializeEngine(engine_file);
        builder->destroy();
    }
    mContext = mEngine->createExecutionContext();
}

void SampleAlexnet::doInference(float* input, float* output) {
    assert(mEngine->getNbBindings() == 2);
    void* buffers[2];

    int input_idx = mEngine->getBindingIndex(INPUT_BLOB_NAME);
    int output_idx = mEngine->getBindingIndex(OUTPUT_BLOB_NAME);

    CUDA_CHECK(cudaMalloc(&buffers[input_idx], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[output_idx], BATCH_SIZE * NUM_CLASSES * sizeof(float)));

    // using cuda stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // copy input data to device and do inference, then copy output from device to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[input_idx], input, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    mContext->enqueue(BATCH_SIZE, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[output_idx], BATCH_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release source
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[input_idx]));
    CUDA_CHECK(cudaFree(buffers[output_idx]));
}
