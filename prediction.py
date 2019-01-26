#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ahmernajar

"""

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

img_path = 'b3.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = classifier.predict(x)
print(preds)
