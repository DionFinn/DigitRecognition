import sys
import os
from preproccess.handler import segmentImage
import tensorflow as tf
from tensorflow import keras
import numpy as np

try:
    model = keras.models.load_model("./models/SEQ/sequential.keras")
    print("model loaded!")
except Exception as e:
    print("model not loaded " + e)
    sys.exit(1)

image = "./images/24.png"
if image is None:
    print("image path not found")
    sys.exit(1)

result = segmentImage(image)
print(result)
    


