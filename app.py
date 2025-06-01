# gui_app.py

import tkinter as tk
from PIL import ImageGrab, Image
import joblib
import numpy as np

# Load trained model
model = joblib.load("digit_model.pkl")

def predict_digit(image):
    # Resize to 8x8, convert to grayscale
    image = image.resize((8, 8)).convert('L')
    image_np = np.array(image)

    # Invert colors (black on white → white on black)
    image_np = 255 - image_np

    # Scale to 0–16 like original dataset
    image_np = (image_np / 255.0) * 16

    # Flatten and predict
    image_np = image_np.flatten().reshape(1, -1)
    return model.predict(image_np)[0]

def clear_canvas():
    canvas.delete("all")
    result_label.config(text="Draw a digit and click Predict")

def on_mouse_drag(event):
    x, y = event.x, event.y
    r = 10  # Brush size
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")

def predict():
    # Capture canvas content
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    img = ImageGrab.grab().crop((x, y, x1, y1))
    digit = predict_digit(img)
    result_label.config(text=f"Prediction: {digit}")

# GUI setup
root = tk.Tk()
root.title("Handwritten Digit Recognizer")

canvas = tk.Canvas(root, width=200, height=200, bg='white')
canvas.pack()
canvas.bind("<B1-Motion>", on_mouse_drag)

tk.Button(root, text="Predict", command=predict).pack()
tk.Button(root, text="Clear", command=clear_canvas).pack()

result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Helvetica", 14))
result_label.pack()

root.mainloop()
