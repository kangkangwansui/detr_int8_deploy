#ifndef BUILDENGINE_H
#define BUILDENGINE_H

#include "zkCommon.h"

using samplesCommon::SampleUniquePtr;

class zkOnnxDeployInt8
{
public:
    zkOnnxDeployInt8(const InputParams& params): mParams(params), mEngine(nullptr){
    }

    bool build(DataType dataType);

    bool infer();

    // bool teardown();

private:
    InputParams mParams;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser, DataType dataType);
};

#endif // BUILDENGINE_H