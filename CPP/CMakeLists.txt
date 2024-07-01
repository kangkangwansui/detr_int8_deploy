cmake_minimum_required(VERSION 3.16)

project(INT8 LANGUAGES CXX)

file(GLOB SAMPLES_COMMON_SOURCES "/home/zhoukang/DevelopToolKit/TensorRT-8.2.1.8/samples/common/logger.cpp")

include_directories(/home/zhoukang/DevelopToolKit/TensorRT-8.2.1.8/include)
include_directories(/home/zhoukang/DevelopToolKit/yaml-cpp-master/include)
include_directories(/home/zhoukang/DevelopToolKit/TensorRT-8.2.1.8/samples/common)
include_directories(/home/zhoukang/DevelopToolKit/opencv-4.8.0/opencv-4.8.0-release/include/opencv4)


link_directories(/usr/local/cuda-11.0/targets/x86_64-linux/lib)
link_directories(/home/zhoukang/DevelopToolKit/yaml-cpp-master/build)
link_directories(/home/zhoukang/DevelopToolKit/TensorRT-8.2.1.8/lib)
link_directories(/home/zhoukang/DevelopToolKit/opencv-4.8.0/opencv-4.8.0-release/lib)

file(GLOB_RECURSE LIBRARIES_opencv "/home/zhoukang/DevelopToolKit/opencv-4.8.0/opencv-4.8.0-release/lib/lib*.so")
file(GLOB_RECURSE LIBRARIES_tensorrt "/home/zhoukang/DevelopToolKit/TensorRT-8.2.1.8/lib/lib*.so")
file(GLOB_RECURSE LIBRARIES_cuda "/usr/local/cuda-11.0/targets/x86_64-linux/lib/lib*.so")
file(GLOB_RECURSE LIBRARIES_yaml_cpp "/home/zhoukang/DevelopToolKit/yaml-cpp-master/build/libyaml-cpp.a")

add_executable(INT8 zkCalibrator.cpp zkutils.cpp zkOnnxDeployInt8.cpp ${SAMPLES_COMMON_SOURCES})

target_link_libraries(INT8 ${LIBRARIES_opencv})
target_link_libraries(INT8 ${LIBRARIES_tensorrt})
target_link_libraries(INT8 ${LIBRARIES_cuda})
target_link_libraries(INT8 ${LIBRARIES_yaml_cpp})