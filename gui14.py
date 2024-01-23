import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image as keras_image

def DrowsyModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect(file_path):
    global label1, sign_image, photo_image

    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    drowsy_detected = detect_drowsiness(eyes)

    for (x, y, w, h) in eyes:
        color = (0, 0, 255)  # Red color for eyes
        if drowsy_detected:
            color = (0, 255, 0)  # Green color for drowsiness detection
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # Perform drowsiness detection using the model
    if drowsy_detected:
        label = "Drowsy"
    else:
        label = "Not Drowsy"

    # Add text to the image
    cv2.putText(image, label, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    # Resize the image to fit the label
    target_width = 300
    target_height = 225
    image = image.resize((target_width, target_height))

    photo_image = ImageTk.PhotoImage(image)

    # Adjusted placement of widgets
    sign_image.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    sign_image.configure(image=photo_image)
    sign_image.image = photo_image
    label1.place(relx=0.95, rely=0.95, anchor=tk.SE)  # Adjusted placement
    label1.configure(text=label)

def show_detect_button(file_path):
    detect_b = Button(top, text="Detect Drowsiness", command=lambda: detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        global photo_image
        file_path = filedialog.askopenfilename()
        print("Selected file path:", file_path)

        uploaded = Image.open(file_path)

        # Set a fixed size for the displayed image
        target_width = 300
        target_height = 225

        # Resize the image to fit the label
        uploaded = uploaded.resize((target_width, target_height))
        photo_image = ImageTk.PhotoImage(uploaded)

        # Adjusted placement of widgets
        sign_image.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        sign_image.configure(image=photo_image)
        sign_image.image = photo_image
        label1.place(relx=0.95, rely=0.95, anchor=tk.SE)  # Adjusted placement
        label1.configure(text='')

        show_detect_button(file_path)
    except Exception as e:
        print("Error:", str(e))

def detect_drowsiness(eyes):
    drowsiness_threshold = 2
    return len(eyes) < drowsiness_threshold

label1 = None
sign_image = None
photo_image = None

top = tk.Tk()
top.geometry('800x600')
top.title('Drowsiness Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
model = DrowsyModel("model.json", "model_weights.h5")

heading = Label(top, text='Drowsiness Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

sign_image.pack(side='left', padx=20)  # Adjusted placement to the left
label1.pack(side='left', padx=10)  # Adjusted placement to the left

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

top.mainloop()
