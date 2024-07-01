#include "include/zkCalibrator.h"
#include "include/zkutils.h"

ZkInt8Calibrator2::ZkInt8Calibrator2(int batchsize,int input_w,int input_h,const char* img_dir,
                                        const char* calib_table_name,const char* input_blob_name,bool read_cache)
    :_batchsize(batchsize)
    ,_input_w(input_w)
    ,_input_h(input_h)
    ,_img_dir(img_dir)
    ,_calib_table_name(calib_table_name)
    ,_input_blob_name(input_blob_name)
    ,_read_cache(read_cache){
    _input_count = _batchsize * _input_w * _input_h * 3;
    cudaMalloc(&_device_input,_input_count * sizeof(float));
    read_files_in_dir(img_dir, _img_files);
}

ZkInt8Calibrator2::~ZkInt8Calibrator2(){
    cudaFree(_device_input);
}

int ZkInt8Calibrator2::getBatchSize() const noexcept{
    return _batchsize;
}

bool ZkInt8Calibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept{
    if (_img_idx + _batchsize > (int)_img_files.size()) {
        return false;
    }

    std::vector<cv::Mat> _input_imgs;
    for (int i = _img_idx; i < _img_idx + _batchsize; i++) {
        std::cout << _img_files[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(_img_dir + "/" + _img_files[i]);
        if (temp.empty()){
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        cv::Mat pr_img = preprocess_img(temp, _input_w, _input_h);
        _input_imgs.push_back(pr_img);
    }
    _img_idx += _batchsize;
    cv::Mat blob = cv::dnn::blobFromImages(_input_imgs, 1.0 / 255.0, cv::Size(_input_w, _input_h), cv::Scalar(0, 0, 0), true, false);
    cudaMemcpy(_device_input, blob.ptr<float>(0), _input_count * sizeof(float), cudaMemcpyHostToDevice);
    bindings[0] = _device_input;
    return true;
}

const void* ZkInt8Calibrator2::readCalibrationCache(size_t& length) noexcept{
    std::cout << "reading calib cache: " << _calib_table_name << std::endl;
    _calib_cache.clear();
    std::ifstream input(_calib_table_name, std::ios::binary);
    input >> std::noskipws;
    if (_read_cache && input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(_calib_cache));
    }
    length = _calib_cache.size();
    return length ? _calib_cache.data() : nullptr;
}

void ZkInt8Calibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept{
    std::cout << "writing calib cache: " << _calib_table_name << " size: " << length << std::endl;
    std::ofstream output(_calib_table_name, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

