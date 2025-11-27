import tkinter as tk
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image
import numpy as np
from tensorflow import keras
from tkinter import messagebox
from preprocess.handler import segmentImage

try:
    model = keras.models.load_model(("./models/SEQ/sequential.keras"))
except Exception as e:
    print("failed to load model", e)
    sys.exit(1)

root = tk.Tk()
root.title("v1 keras model")

canvas = tk.Canvas(root, bg="white")
canvas.pack(fill="both", expand=True)

is_drawing = False
drawing_color = "black"
line_width = 50

controls_frame = tk.Frame(root)
controls_frame.pack(side="top", fill="x")


clear_button = tk.Button(controls_frame, text="Clear Canvas", command=lambda: canvas.delete("all"))
clear_button.pack(side="left", padx=5, pady=5)


root.geometry("28x28")
def start_drawing(event):
    global is_drawing, prev_x, prev_y
    is_drawing = True
    prev_x, prev_y = event.x, event.y

def draw(event):
    global is_drawing, prev_x, prev_y
    if is_drawing:
        current_x, current_y = event.x, event.y
        canvas.create_line(prev_x, prev_y, current_x, current_y, fill=drawing_color, width=line_width, capstyle=tk.ROUND, smooth=True)
        prev_x, prev_y = current_x, current_y

def stop_drawing(event):
    global is_drawing
    is_drawing = False

def change_line_width(value):
    global line_width
    line_width = int(value)

def submit():
    image_path = "images/"
    if os.path.exists(image_path):
        pass
    else:
        os.makedirs(image_path,  exist_ok=True)

    temp_ps = "runtime_canvas.ps"
    temp_png = "runtime_canvas.png"

    canvas.postscript(file=temp_ps, colormode="color")

    image = Image.open(temp_ps).convert("L")
    image.save(temp_png)
    
    try:
        result = segmentImage(temp_png)
        messagebox.showinfo("Prediction", f"Predicted digit: {result}")
        return result
    except Exception as e:
        messagebox.showinfo("Submission Failed")
        print("error with submission ", e)
        sys.exit(1)
    
    
submit_button = tk.Button(controls_frame, text="Submit Canvas", command=submit)
submit_button.pack(side="left", padx=5, pady=5)



canvas.bind("<ButtonPress-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)


root.mainloop()



    # filecount = len(os.listdir("images"))
    # temp_ps = f"images/canvas{filecount}.ps"
    # temp_png = f"images/canvas{filecount}.png"