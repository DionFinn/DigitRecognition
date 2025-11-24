import sys
import os
from preproccess.handler import imageHandler
import tensorflow as tf
from tensorflow import keras
import numpy as np

try:
    model = keras.models.load_model("./models/SEQ/sequential.keras")
    print("model loaded!")
except Exception as e:
    print("model not loaded " + e)
    sys.exit(1)

image = imageHandler("./images/DionDrawing2.png")
if image is None:
    ValueError("Error loading image, check path")    

prediction = model.predict(image)
predicted_class = int(np.argmax(prediction, axis=1)[0])
print(f"Predicted digit: {predicted_class}")

    


