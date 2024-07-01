import os
import onnx
import glob
import cv2

import numpy as np
import onnxruntime as rt

from collections import OrderedDict
from utils import cvtColor,preprocess_input

class MakeDynamicRange:
    def __init__(self,img_dir,onnx_path,save_dynamic_range_path,length=800):
        self.img_dir = img_dir
        self.onnx_path = onnx_path
        self.save_dynamic_range_path = save_dynamic_range_path
        self.length = length
        self.model = self.Init_model()
        self.providers = ['CUDAExecutionProvider']
        self.Init_dynamic_range = self.Init_value_of_dynamic_range()
        self.ort_session = rt.InferenceSession(self.model.SerializeToString(), providers=self.providers)
    
    def Init_model(self):
        model = onnx.load(self.onnx_path)
        for node in model.graph.node:
            for output in node.output:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])  #将名为“output”的输出添加到模型的输出列表中
        return model
        

    def Init_value_of_dynamic_range(self):  #image_shape is (batch,channel,height,width)
        Init_dynamic_range = {}
        for node in self.model.graph.node:
            for output in node.output:
                Init_dynamic_range[output] = 0
        return Init_dynamic_range
    
    def get_layer_abs_max(self,ort_outs):
        layers_abs_max = {}
        for key in ort_outs.keys():
            layers_abs_max[key] = np.max(abs(ort_outs[key]))    
        return layers_abs_max

    def tran_pictrue(self,image_path):
        # 读取图片
        image = cv2.imread(image_path)

        image = cvtColor(image)

        ih, iw  = image.shape[:2]

        scale = min(self.length/iw, self.length/ih)

        nw = int(iw*scale)
        nh = int(ih*scale)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = np.zeros((self.length, self.length, 3), dtype=np.uint8) + (128, 128, 128)

        offset_x = (self.length - nw) // 2
        offset_y = (self.length - nh) // 2

        new_image[offset_y:offset_y + nh, offset_x:offset_x + nw] = image

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(new_image, dtype='float32')), (2, 0, 1)), 0)

        return image_data
 
    def get_layer_outputs(self,im):
        outputs_name = []
    
        input_name = self.ort_session.get_inputs()[0].name
        
        outputs_name = [x.name for x in self.ort_session.get_outputs()]

        outputs = self.ort_session.run(outputs_name, {input_name: im})

        ort_outs = OrderedDict(zip(outputs_name, outputs))

        # for key,value in ort_outs.items():
        #     print(f"Key: {key}, Value: {value}")
        #     print(" ")
        
        return ort_outs

    def make_dynamic_range(self):
        imgs_path_list = glob.glob(os.path.join(self.img_dir, "*"))
        tatolly = len(imgs_path_list)
        for index,image_path in enumerate(imgs_path_list):
            img = self.tran_pictrue(image_path)
            output = self.get_layer_outputs(img)
            layers_abs_max = self.get_layer_abs_max(output)

            for key in layers_abs_max.keys():
                self.Init_dynamic_range[key] = max(self.Init_dynamic_range[key],layers_abs_max[key])         

            if index % 20 == 0:
                print(f"Process: {index}/{tatolly}")
                #保存每个层输出的动态最大值
                with open(self.save_dynamic_range_path, 'w') as f:
                    for key,value in self.Init_dynamic_range.items():
                        f.write(f"{key}:{value}\n")
        
        with open(self.save_dynamic_range_path, 'w') as f:
            for key,value in self.Init_dynamic_range.items():
                f.write(f"{key}:{value}\n")
        
        # for key,value in init_dynamic_range.items():
        #     print(f"Key: {key}, Value: {value}")
        #     print(" ")


if __name__ == '__main__':
    image_dir = '/home/zhoukang/quantCode/zkTest/tensorrt_test/calibration_data'

    onnx_path = '/home/zhoukang/quantCode/zkTest/tensorrt_test/model_data/detr.onnx'

    save_dynamic_range_path = "/home/zhoukang/quantCode/zkTest/tensorrt_test/model_data/detr_dynamic_range.txt"

    image_shape = (1, 3, 800, 800)

    length = 800

    print()

    makedunamicrange = MakeDynamicRange(image_dir,onnx_path,save_dynamic_range_path,length)
    makedunamicrange.make_dynamic_range()
    
