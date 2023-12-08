# Import necessary libraries
import customtkinter as ctk
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine
import time
from anti_spoofing import AntiSpoofingSystem

# Set up interface colors, fonts, and styles
primary_color = "#007BFF"  # Primary color (blue)
secondary_color = "#FFFFFF"  # Secondary color (white)
text_color = "#333333"  # Text color (dark)

# Instantiate the AntiSpoofingSystem
anti_spoofing_system = AntiSpoofingSystem()

# Define button dimensions and styles
button_width, button_height, font_size, font_size_total = 160, 40, 16, 20
corner_radius, border_color, border_width = 8, primary_color, 2

# Function to create a styled button
def create_button(parent, text, command):
    return ctk.CTkButton(parent, text=text, command=command, fg_color=primary_color,
                         text_color=secondary_color, hover_color=text_color, font=("Roboto", font_size),
                         width=button_width, height=button_height,
                         corner_radius=corner_radius, border_color=border_color, border_width=border_width)

# Directories for saving registered images and captured images
register_directory = "register"
person_directory = "Person"
if not os.path.exists(register_directory):
    os.makedirs(register_directory)
if not os.path.exists(person_directory):
    os.makedirs(person_directory)

global register_fill  # Global variable for user input field
register_fill = None

# Load the TensorFlow model for face recognition
RECOGNITION_THRESHOLD = 0.04
embedding_model = tf.keras.models.load_model('best_model.h5')

# Preprocess an image for the model
def preprocess_image(image):
    image = cv2.resize(image, (375, 375))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Generate an embedding for an image using the model
def generate_embedding(image):
    preprocessed_image = preprocess_image(image)
    return embedding_model.predict(preprocessed_image)[0]

# Check if a user is already registered by comparing embeddings
def is_user_already_registered(captured_frame):
    try:
        captured_embedding = generate_embedding(captured_frame)
        for filename in os.listdir(register_directory):
            if filename.endswith(".png"):
                registered_image = cv2.imread(os.path.join(register_directory, filename))
                if registered_image is not None:
                    registered_embedding = generate_embedding(registered_image)
                    distance = cosine(captured_embedding, registered_embedding)
                    if distance <= RECOGNITION_THRESHOLD:
                        return True
        return False
    except Exception as e:
        print(f"Error in user existence check: {str(e)}")
        return False

# Interface logic for user registration
def register_interface():
    global register_fill
    user_id = register_fill.get()
    if not user_id:
        status_label.configure(text="User ID is empty")
        return

    frame = anti_spoofing_system.get_captured_image()
    if frame is not None:
        if is_user_already_registered(frame):
            status_label.configure(text="User already exists.")
        else:
            image_filename = os.path.join(register_directory, f"{user_id}.png")
            cv2.imwrite(image_filename, frame)
            status_label.configure(text=f"User {user_id} registered successfully.")
            reset_anti_spoofing()
    else:
        status_label.configure(text="No frame captured.")

# Set for tracking checked-in students
checked_in_students = set()

# Update the label showing the total number of checked-in students
def update_total_students_label():
    total_students_label.configure(
        text=f"Total Students: {len(checked_in_students)}")

# Interface logic for checking in a user
def check_in():
    global status_label, checked_in_students

    frame = anti_spoofing_system.get_captured_image()
    if frame is None:
        status_label.configure(text="Please complete the anti-spoofing check first.")
        return

    if is_user_already_registered(frame):
        recognized_user_id = get_user_id_from_frame(frame)
        if recognized_user_id:
            if recognized_user_id not in checked_in_students:
                checked_in_students.add(recognized_user_id)
                status_label.configure(text=f"Attendance recorded for {recognized_user_id}.")
                update_total_students_label()
            else:
                status_label.configure(text=f"Attendance already recorded for {recognized_user_id}.")
        else:
            status_label.configure(text="User does not exist in register folder.")
    else:
        status_label.configure(text="User does not exist in register folder.")
    
    reset_anti_spoofing()

# Retrieve the user ID from the frame by comparing embeddings
def get_user_id_from_frame(captured_frame):
    captured_embedding = generate_embedding(captured_frame)
    for filename in os.listdir(register_directory):
        if filename.endswith(".png"):
            user_id = filename.split('.')[0]
            registered_image = cv2.imread(os.path.join(register_directory, filename))
            registered_embedding = generate_embedding(registered_image)
            distance = cosine(captured_embedding, registered_embedding)
            if distance <= RECOGNITION_THRESHOLD:
                return user_id
    return None

# Update the video frame and handle anti-spoofing logic
def update_frame():
    global none_spoofed_image

    frame = anti_spoofing_system.run()
    if frame is not None:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        if anti_spoofing_system.blink_count >= 5 and not anti_spoofing_system.image_captured:
            timestamp = int(time.time())
            image_name = f"captured_{timestamp}.png"
            cv2.imwrite(os.path.join(person_directory, image_name), frame)
            anti_spoofing_system.image_captured = True
            anti_spoofing_system.blink_count = 0  # Reset the blink counter

        if anti_spoofing_system.anti_spoofing_completed:
            verified_label.configure(text="AntiSpoofed: True")
            button_frame.pack(pady=20)
        else:
            verified_label.configure(text="AntiSpoofed: False")
            button_frame.pack_forget()

    video_label.after(10, update_frame)

# Reset anti-spoofing status
def reset_anti_spoofing():
    anti_spoofing_system.anti_spoofing_completed = False
    anti_spoofing_system.blink_count = 0
    anti_spoofing_system.image_captured = False

# Initialize the app and start the main loop
app = ctk.CTk()
app.title("Face Recognition Attendance System")
app.geometry("1440x810")
ctk.set_appearance_mode("light")

cap = cv2.VideoCapture(0)
main_frame = ctk.CTkFrame(app, fg_color=secondary_color)
main_frame.pack(expand=True, fill='both')
video_label = tk.Label(main_frame, width=930, height=650)
video_label.grid(row=0, column=0, padx=10, columnspan=2)

checked_in_students = set()
total_students_label = ctk.CTkLabel(
    main_frame, text=f"Total Students: {len(checked_in_students)}", font=("Roboto", font_size_total))
total_students_label.grid(row=0, column=2, sticky="nw", padx=10)

verified_label = ctk.CTkLabel(
    main_frame, text="AntiSpoofed: False", font=("Roboto", font_size_total))
verified_label.grid(row=0, column=2, sticky="sw", padx=10)

status_label = ctk.CTkLabel(main_frame, text="", font=("Roboto", font_size), fg_color=secondary_color, text_color=text_color)
status_label.grid(row=1, column=0, padx=10, columnspan=2)

button_frame = ctk.CTkFrame(main_frame, fg_color=secondary_color)
button_frame.grid(row=2, column=0, columnspan=3)
button_frame.pack_forget()

buttonRegister = create_button(button_frame, "Register", register_interface)
buttonRegister.pack(side="left", padx=10)

register_fill = ctk.CTkEntry(button_frame)
register_fill.pack(side="left", padx=10)

buttonCheckIn = create_button(button_frame, "Check Database", check_in)
buttonCheckIn.pack(side="left", padx=10)

update_frame()
app.mainloop()
cap.release()
