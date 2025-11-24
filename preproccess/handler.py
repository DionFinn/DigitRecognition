import numpy as np
import tensorflow as tf
import cv2

def imageHandler(imagepath):
    if not imagepath:
        return ValueError("img path not found")
    else:
        image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
        image = 255 - image
        image = image.astype("float32") / 255
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image
