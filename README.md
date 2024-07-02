## 基于tensorRT的DETR目标检测模型INT8量化以及部署
---

## CPP文件目录
1. include ： 项目所需的头文件，zkCommon.h文件包含了所有的头文件。
2. config.yaml :  配置文件，包含了校准器、构建模型，模型推理等阶段的输入参数。
3. CMakeLists.txt ：本项目是利用cmake编译，只需要将CMakeLists.txt文件中相关路径替换成自己的本地路径即可编译成功。
4. zkCalibrator.cpp ： 自定义的校准器，继承自IInt8EntropyCalibrator2，该校准器可以统计每一层输入数据的动态范围，并生成int8calib.table文件
5. zkutils.cpp ： 包含了读取图片路径、图片预处理、softmax等函数
6. zkOnnxDeployInt8.cpp ： 根据onnx构建网络，并设置int8量化配置，生成int8 engine文件，最后利用engine文件进行推理

## 项目结构
```
├── CMakeLists.txt
├── config.yaml
├── zkutils.cpp 
├── zkCalibrator.cpp
├── zkOnnxDeployInt8.cpp
├── include
│   ├── zkCommon.h
│   ├── zkutils.h
│   ├── zkCalibrator.h
│   ├── zkOnnxDeployInt8.h 
```

## 所需环境
c++环境：
cmake == 3.16.3
tensorrt == 8.2.1.8
opencv == 4.8.0
yaml-cpp

## 文件下载
部署模型需要的detr.onnx、校准数据集、int8calib.table、voc_classes.txt等文件，以及序列化生成的int8 engine文件，可在百度网盘中下载。  
链接：https://pan.baidu.com/s/1J2Z1F6tmIRPnTuMubTClcg 
提取码：wj8g

## 编译项目并运行
1. 在编译项目之前需要将配置需要的参数预设好值，INPUT_BLOB_NAME 设置为自己的detr.onnx模型的输入，不清楚的可以通过netron（直接百度）查看；isBuild、isSaveEngine  需要设置为true，表示需要构建模型，生成序列化引擎（否则推理不了）；其他参数按照自己的需求设置。

2. 编译项目
```
cd CPP
cmake -B build
cmake --build build

```

3. 运行项目，生成int8 engine文件
```
./build/INT8

```
int engine引擎会生成在指定的文件夹下，可以直接部署到TensorRT上进行推理。

4. 反序列化engine，推理图片
在运行项目之前，需要将isBuild设置为false，目的是切换项目模式，即取消构建引擎模式，执行推理模式；img_path ：设置为需要推理的图片路径；engineflie设置为生成的int8 engine文件路径。

```
./build/INT8

```






