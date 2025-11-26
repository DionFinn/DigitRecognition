import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras


model = keras.models.load_model("./models/SEQ/sequential.keras")

def processSingleDigit(segment):
    if segment is None or segment.size == 0:
        raise ValueError("Empty digit segment passed to processSingleDigit")

    digit = cv2.resize(segment, (28, 28), interpolation=cv2.INTER_AREA)
    digit = 255 - digit
    digit = digit.astype("float32") / 255.0

    digit = np.expand_dims(digit, axis=-1)
    digit = np.expand_dims(digit, axis=0)

    return digit

def segmentImage(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"path not found: {image_path}")
    
    imageblur = cv2.GaussianBlur(image, (5, 5), 0)

    _, threshold = cv2.threshold(
        imageblur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("no digits found within image")

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours_sorted = [
        c for _, c in sorted(
            zip(bounding_boxes, contours),
            key=lambda b: b[0][0]
        )
    ]

    predicted_digits = []

    for c in contours_sorted:
        x, y, w, h = cv2.boundingRect(c)
        if w < 5 or h < 5:
            continue

        digit_roi = image[y:y+h, x:x+w]

        digit_input = processSingleDigit(digit_roi)

        pred = model.predict(digit_input, verbose=0)
        digit_class = int(np.argmax(pred, axis=1)[0])
        predicted_digits.append(str(digit_class))

    return "".join(predicted_digits)