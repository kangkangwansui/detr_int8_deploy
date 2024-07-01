#ifndef ZKCALIBRATOR_H
#define ZKCALIBRATOR_H

#include "zkCommon.h"

class ZkInt8Calibrator2 : public nvinfer1::IInt8EntropyCalibrator2{
public:
    ZkInt8Calibrator2(int batchsize,int input_w,int input_h,const char* img_dir,
                        const char* calib_table_name,const char* input_blob_name,bool read_cache);
    virtual ~ZkInt8Calibrator2();
    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    int _batchsize;
    int _input_w;
    int _input_h;
    int _img_idx;
    void* _device_input;
    bool _read_cache;
    const char* _input_blob_name;
    size_t _input_count;
    std::string _img_dir;
    std::string _calib_table_name;
    std::vector<std::string> _img_files;
    std::vector<char> _calib_cache;
};


#endif // ENTROPY_CALIBRATOR_H