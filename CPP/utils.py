import cv2
import numpy as np
from PIL import Image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image 
    
def resize_image(image, min_length):
    iw, ih  = image.shape[:2]
    # image = Image.fromarray(image)
    h, w    = get_new_img_size(ih, iw, min_length=min_length)
    new_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    # new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_new_img_size(height, width, min_length=600):
    if width <= height:
        f = float(min_length) / width
        resized_height = int(f * height)
        resized_width = int(min_length)
    else:
        f = float(min_length) / height
        resized_width = int(f * width)
        resized_height = int(min_length)

    return resized_height, resized_width

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image