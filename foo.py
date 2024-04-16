import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

IMG_WIDTH = 256
IMG_HEIGHT = 256

model_paths = {
    "Effusion": "/Users/savraj/programming/X-R-A-I/effusion_model.keras",
    "Atelectasis": "/Users/savraj/programming/X-R-A-I/atelectasis_model.keras",
    "Pneumonia": "/Users/savraj/programming/X-R-A-I/pneumonia_model.keras",
    "Cardiomegaly": "/Users/savraj/programming/X-R-A-I/cardiomegaly_model.keras"
}
models = {name: tf.keras.models.load_model(
    path) for name, path in model_paths.items()}


def load_image_into_numpy_array(image):
    image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def classify_image(filename, model):
    image = Image.open(filename).resize((IMG_HEIGHT, IMG_WIDTH))
    image_np = load_image_into_numpy_array(image)
    exp = np.true_divide(image_np, 255.0)
    expanded = np.expand_dims(exp, axis=0)
    return model.predict(expanded)[0][0]


def predict_img(filename, model_name):
    confidence = classify_image(filename, models[model_name])
    guess = 'positive' if confidence > 0.5 else 'negative'
    return guess, confidence


window = tk.Tk()
window.title("X-R-AI")
window.geometry("600x500")


frame = tk.Frame(window, padx=20, pady=20)
frame.place(relx=0.5, rely=0.5, anchor="center")


title_label = tk.Label(frame, text="X-R-AI", font=("Helvetica", 16))
title_label.grid(row=0, column=0, columnspan=2)


result_label = tk.Label(frame, text="", font=("Helvetica", 12), wraplength=500)
result_label.grid(row=3, column=0, columnspan=2, pady=(20, 0))


image_label = tk.Label(frame)
image_label.grid(row=1, column=0, columnspan=2)


selected_model = tk.StringVar()
model_options = list(model_paths.keys())
model_dropdown = tk.OptionMenu(frame, selected_model, *model_options)
selected_model.set(model_options[0])
model_dropdown.grid(row=2, column=0, pady=10)


def browse_and_classify():
    file_path = filedialog.askopenfilename()
    if file_path:
        model_name = selected_model.get()

        guess, confidence = predict_img(file_path, model_name)

        result_label.config(
            text=f"Prediction ({model_name}): {guess} (Confidence: {confidence:.2f})")

        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo


browse_button = tk.Button(frame, text="Browse Image",
                          command=browse_and_classify)
browse_button.grid(row=2, column=1, pady=10)

window.mainloop()
