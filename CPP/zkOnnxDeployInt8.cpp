#include "include/zkOnnxDeployInt8.h"
#include "include/zkCalibrator.h"
#include "include/zkutils.h"

int maxBatchSize = 1;

bool zkOnnxDeployInt8::build(DataType dataType){
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        std::cout << "Failed to create builder!" << std::endl;
        return false;
    }

    if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8())
        || (dataType == DataType::kHALF && !builder->platformHasFastFp16()))
    {
        std::cout << "Int8 quantization is not supported!" << std::endl;  
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cout << "Failed to create network!" << std::endl;
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cout << "Failed to create config!" << std::endl;
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        std::cout << "Failed to create parser!" << std::endl;
        return false;
    }

    if(!constructNetwork(builder, network, config, parser, dataType)){
        std::cout << "Failed to construct network!" << std::endl;
        return false;
    }

    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        std::cout << " fail in profileStream " << std::endl;
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        std::cout << " fail in plan " << std::endl;
        return false;
    }

    if(mParams.isSaveEngine){
        std::string engine_output_path = mParams.EngineOutputFile + '/' + mParams.modelname + "_int8.engine";
        std::ofstream p(engine_output_path, std::ios::binary);
        if (!p){
            std::cout << "could not open engine_output_path create this file" << std::endl;
            std::ofstream new_file(engine_output_path.c_str(), std::ios::out | std::ios::binary);
            if (new_file.is_open()) {
                std::cout << engine_output_path << " has been created." << std::endl;
            } else {
            std::cerr << "Failed to create " << engine_output_path << "." << std::endl;
            }
        }  
        p.write(reinterpret_cast<const char*>(plan->data()), plan->size());
        std::cout<< "引擎已经保存在：" << engine_output_path << std::endl;
    }

    return true;
}

bool zkOnnxDeployInt8::infer(){
    auto beforeTime = std::chrono::steady_clock::now();
    std::vector<OutputParam> boxes_information;
    cv::Mat img = cv::imread(mParams.img_path);
    cv::Mat rgbImage;
    if (img.channels() == 3) {
        rgbImage = img.clone(); 
    } else {
        // 将图像转换为RGB格式
        cv::cvtColor(img, rgbImage, cv::COLOR_BGR2RGB);
    }

    float *input_blob = new float[mParams.modelLenth * mParams.modelLenth * 3];
    //图像预处理
    cv::Mat precesses_img = preprocess_img(rgbImage, mParams.modelLenth, mParams.modelLenth);

    //将输入图片转为数据格式
    const int channels = precesses_img.channels();
    const int width = precesses_img.cols;
    const int height = precesses_img.rows;

    std::cout << "Hight is : " << height << "," << " Width is : " << width << std::endl;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                input_blob[c * width * height + h * width + w] = precesses_img.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }

    std::string engineflie = mParams.engineflie;
    std::ifstream file(engineflie,std::ios::binary);
    if(!file.good()){
        std::cout << "引擎加载失败" << std::endl;
        return false;
    }
    size_t size = 0;
    file.seekg(0, file.end);    //将读指针从文件末尾开始移动0个字节
    size = file.tellg();        //获取读指针的位置，即文件末尾的字节数

    if (size == 0) {
        std::cout << "引擎文件是空的" << std::endl;
        return false;
    }

    file.seekg(0, file.beg);    //将读指针从文件开头开始移动0个字节
    char* TRTmodelStream = new char[size];
    file.read(TRTmodelStream, size);
    file.close();

    auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    auto engine = SampleUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(TRTmodelStream, size, nullptr));
    delete[] TRTmodelStream;
    if (engine == nullptr) {
        return false;
    }

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    void *buffers[3];

    nvinfer1::Dims input_dim = engine->getBindingDimensions(0);
    std::cout << "input_dim.nbDims: " << input_dim.nbDims << std::endl;
    int input_size = 1;
    for(int j = 0; j < input_dim.nbDims; j++){
        input_size *= input_dim.d[j];
    }
    std::cout << "input_size: " << input_size << std::endl;
    cudaMalloc(&buffers[0], input_size * sizeof(float));

    nvinfer1::Dims output_dim_1 = engine->getBindingDimensions(1);
    int output_size_1 = 1;
    for(int i = 0;i < output_dim_1.nbDims; i++){
        output_size_1 *= output_dim_1.d[i];
    }
    cudaMalloc(&buffers[1], output_size_1 * sizeof(float));
    std::cout << "output_size_1: " << output_size_1 << std::endl;

    nvinfer1::Dims output_dim_2 = engine->getBindingDimensions(2);
    int output_size_2 = 1;
    for(int i = 0;i < output_dim_2.nbDims; i++){
        output_size_2 *= output_dim_2.d[i];
    }
    cudaMalloc(&buffers[2], output_size_2 * sizeof(float));
    std::cout << "output_size_2: " << output_size_2 << std::endl;

    float *output_CpuBuffer_1 = new float[output_size_1]();
    float *output_CpuBuffer_2 = new float[output_size_2]();  // 给输出分配cpu内存，以接收GPU计算的结果

    cudaStream_t stream;
    cudaStreamCreate(&stream);  //在GPU创建进程束

    // 将输入数据从CPU传输到GPU
    cudaMemcpy(buffers[0], input_blob, input_size * sizeof(float), cudaMemcpyHostToDevice);  

    context->enqueueV2(buffers, stream, nullptr);

    cudaMemcpy(output_CpuBuffer_1, buffers[1],output_size_1 * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(output_CpuBuffer_2, buffers[2],output_size_2 * sizeof(float),cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(stream);//等待输出数据传输完毕

    int output_softmax_size = output_size_1;
    float* output_softmax_1 = new float[output_softmax_size]();
    for(int i = 0; i < output_softmax_size; i++){
        output_softmax_1[i] = 0;
    }
    softmax(output_CpuBuffer_1,output_softmax_1,mParams.num,mParams.classes);

    get_boxes_information(output_softmax_1,output_CpuBuffer_2,mParams.confidence,mParams.num,mParams.classes,
                                mParams.classes_txt,boxes_information);

    std::cout << "the boxes_information size is : " << boxes_information.size() << std::endl;

    print_information(boxes_information);

    auto afterTime = std::chrono::steady_clock::now();

    //毫秒级
	double duration_millsecond = std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();
	std::cout << duration_millsecond << "毫秒" << std::endl;

    // 释放资源
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaStreamDestroy(stream);

    delete[] input_blob;
    delete[] output_CpuBuffer_1;
    delete[] output_CpuBuffer_2;
    delete[] output_softmax_1;

    return true;

}

bool zkOnnxDeployInt8::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser, DataType dataType)
{
    mEngine = nullptr;
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;
    auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str(),verbosity);
    if (!parsed)
    {
        std::cout << "Failed to parse ONNX file!" << std::endl;
        return false;
    }

    // Configure buider
    config->setAvgTimingIterations(1);
    config->setMinTimingIterations(1);
    config->setMaxWorkspaceSize(1_GiB);

    builder->setMaxBatchSize(maxBatchSize);

    if(mParams.dataType == DataType::kHALF){
        config->setFlag(BuilderFlag::kFP16);
    }
    if(mParams.dataType == DataType::kINT8){
        nvinfer1::IInt8EntropyCalibrator2 *calibrator = 
            new ZkInt8Calibrator2(mParams.calibParams.batchSize,mParams.calibParams.INPUT_W,
                                mParams.calibParams.INPUT_H,mParams.calibParams.imgDir.c_str(),
                                mParams.calibParams.calibFile.c_str(),mParams.calibParams.INPUT_BLOB_NAME.c_str(),
                                mParams.calibParams.read_cache);
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator);
    }       

    return true;
}

void interface(std::string yamlPath){
    std::cout << "welcome to use DETR !" << std::endl;
    YAML::Node config = YAML::LoadFile(yamlPath);
    InputParams input;
    bool build_flag;
    bool infer_flag;

    input.calibParams.INPUT_W = config["INPUT_W"].as<int>();
    input.calibParams.INPUT_H = config["INPUT_H"].as<int>();
    input.calibParams.batchSize = config["batchSize"].as<int>();
    input.calibParams.read_cache = config["read_cache"].as<bool>();
    input.calibParams.imgDir = config["imgDir"].as<std::string>();
    input.calibParams.calibFile = config["calibFile"].as<std::string>();
    input.calibParams.INPUT_BLOB_NAME = config["INPUT_BLOB_NAME"].as<std::string>();

    input.isBuild = config["isBuild"].as<bool>();
    input.verbose = config["verbose"].as<bool>();
    input.isSaveEngine = config["isSaveEngine"].as<bool>();
    input.num = config["num"].as<int>();
    input.classes = config["classes"].as<int>();
    input.maxBatchSize = config["maxBatchSize"].as<int>();
    input.modelLenth = config["modelLenth"].as<int>();
    input.confidence = config["confidence"].as<float>();
    input.img_path = config["img_path"].as<std::string>();
    input.modelname = config["modelname"].as<std::string>();
    input.engineflie = config["engineflie"].as<std::string>();
    input.classes_txt = config["classes_txt"].as<std::string>();
    input.onnxFileName = config["onnxFileName"].as<std::string>();
    input.dataType = get_data_type(config["dataType"].as<std::string>());
    input.EngineOutputFile = config["EngineOutputFile"].as<std::string>();
    

    zkOnnxDeployInt8 onnxDeployInt8(input);

    if(input.isBuild){
        build_flag = onnxDeployInt8.build(input.dataType);
        std::cout << "build_flag: " << build_flag << std::endl;
    } else{
        infer_flag = onnxDeployInt8.infer();
        std::cout << "infer_flag: " << infer_flag << std::endl;
    }
}


int main(){
    std::string yamlPath = "/home/zhoukang/GithubProject/detr_int8_deploy/CPP/config.yaml";
    interface(yamlPath);
    return 0;
}