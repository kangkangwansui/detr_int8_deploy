#include "include/zkutils.h"

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

DataType get_data_type(std::string data_type){
    if (data_type == "kFLOAT"){
        return DataType::kFLOAT;
    } else if (data_type == "kHALF"){
        return DataType::kHALF;
    } else if (data_type == "kINT8"){
        return DataType::kINT8;
    } else if (data_type == "kINT32"){
        return DataType::kINT32;
    } else {
        return DataType::kBOOL;
    }
}

void softmax(const float* input, float* output, int num, int classes){  
    for (int i = 0; i < num; ++i) {
        float max_val = -INFINITY; // 找到当前框的最大值
        for (int c = 0; c < classes; ++c) {
            max_val = fmaxf(max_val, input[i * classes + c]);
        }

        float sum_exp = 0.0f; // 计算指数和
        for (int c = 0; c < classes; ++c) {
            output[i * classes + c] = expf(input[i * classes + c] - max_val);
            sum_exp += output[i * classes + c];
        }

        // 归一化
        for (int c = 0; c < classes; ++c) {
            output[i * classes + c] /= sum_exp;
        }
    }
}

std::vector<std::string> get_classes_vector(std::string classes_txt){
    std::vector<std::string> classes;
    std::ifstream file(classes_txt);
    if (!file.is_open()) {
        std::cerr << "Unable to open file!" << std::endl;
    }

    std::string line;
    // 按行读取文件
    while (std::getline(file, line)) {
        // 处理每一行
        classes.push_back(line);
    }
    file.close();
    return classes;
}

void get_boxes_information(float* pred_logits,float* pred_boxes,float confidence,int num,int classes,
                                std::string classes_txt,std::vector<OutputParam>& boxes_information){
    std::vector<std::string> classes_vector = get_classes_vector(classes_txt);
    OutputParam box_information;
    for (int i = 0; i < num; ++i){
        for (int c = 0; c < classes - 1; ++c){
            if(pred_logits[i*classes + c] > confidence){
                box_information.box_idx = i;
                box_information.classes = classes_vector[c];
                box_information.score = pred_logits[i*classes + c];
                box_information.center_x = pred_boxes[i * 4];
                box_information.center_y = pred_boxes[i * 4 + 1];
                box_information.width = pred_boxes[i * 4 + 2];
                box_information.height = pred_boxes[i * 4 + 3];
                boxes_information.push_back(box_information);
                break;
            }
        }
    }
}

void print_information(std::vector<OutputParam>& boxes_information){
    for (const auto& param : boxes_information) {
            // 访问每个OutputParam对象的成员变量
            std::cout << "Box Index: " << param.box_idx << std::endl;
            std::cout << "Classes: " << param.classes << std::endl;
            std::cout << "Score: " << param.score << std::endl;
            std::cout << "Center X: " << param.center_x * 255 << std::endl;
            std::cout << "Center Y: " << param.center_y * 255  << std::endl;
            std::cout << "Width: " << param.width * 255  << std::endl;
            std::cout << "Height: " << param.height * 255  << std::endl;
            // 输出换行，为了更好的可读性
            std::cout << std::endl;
        }
}


// cv::Mat preprocess_img(cv::Mat& img,int img_Lenth) {
//     cv::Mat rgbImage;
//     if (img.channels() == 3) {
//         rgbImage = img.clone(); 
//     } else {
//         // 将图像转换为RGB格式
//         cv::cvtColor(img, rgbImage, cv::COLOR_BGR2RGB);
//     }

//     cv::Size original_size = img.size();
//     int original_height = original_size.height;
//     int original_width = original_size.width;

//     double scale = std::min(static_cast<double>(img_Lenth) / original_width, static_cast<double>(img_Lenth) / original_height);

//     int nw = static_cast<int>(original_width * scale);
//     int nh = static_cast<int>(original_height * scale);

//     cv::Mat resized_image;
//     cv::resize(img, resized_image, cv::Size(nw, nh), 0, 0, cv::INTER_CUBIC);

//     cv::Mat new_image(cv::Size(img_Lenth, img_Lenth), img.type(), cv::Scalar(128, 128, 128));

//     int offset_x = (img_Lenth - nw) / 2;
//     int offset_y = (img_Lenth - nh) / 2;

//     cv::Mat roi = new_image(cv::Rect(offset_x, offset_y, nw, nh));
//     resized_image.copyTo(roi);
//     return new_image;
// }