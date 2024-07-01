#ifndef ZKCOMMON_H
#define ZKCOMMON_H

#include "common.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <dirent.h>
#include <fstream>
#include <iterator>
#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

struct CalibratorParams{
    int INPUT_W;
    int INPUT_H;
    int batchSize{1};
    bool read_cache{true};

    std::string imgDir;
    std::string calibFile;
    std::string INPUT_BLOB_NAME;
};

struct InputParams{
    bool isBuild{true};
    bool verbose{false};
    bool isSaveEngine{true};
    int maxBatchSize{1};
    int modelLenth{800};
    int num{100};
    int classes{21};
    float confidence{0.5};

    std::string img_path;
    std::string modelname;
    std::string engineflie;
    std::string classes_txt;
    std::string onnxFileName;
    std::string EngineOutputFile;
    DataType dataType;

    CalibratorParams calibParams;
};

struct OutputParam
{
    int box_idx;
    
    float score;
    float center_x;
    float center_y;
    float width;
    float height;

    std::string classes;
};

#endif // ZKCOMMON_H