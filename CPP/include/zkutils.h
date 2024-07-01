#ifndef ZKUTILS_H
#define ZKUTILS_H

#include "zkCommon.h"

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);

cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);

DataType get_data_type(std::string data_type);

void softmax(const float* input, float* output, int num, int classes);

void get_boxes_information(float* pred_logits,float* pred_boxes,float confidence,int num,int classes,
                                std::string classes_txt,std::vector<OutputParam>& boxes_information);

void print_information(std::vector<OutputParam>& boxes_information);

#endif // ZKUTILS_H