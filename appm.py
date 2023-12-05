import customtkinter as ctk
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import subprocess
from customtkinter import FontManager
import os

# Import your AntiSpoofingSystem here
from anti_spoofing import AntiSpoofingSystem

# Import TensorFlow and other dependencies if needed for the face recognition part
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine

################### INTERFACE ###################

# New Color Scheme
primary_color = "#007BFF"  # Blue
secondary_color = "#FFFFFF"  # White
text_color = "#333333"  # Dark text for readability

# Custom Font
FontManager.load_font("Roboto-Regular.ttf")  # Use Roboto font

# Instantiate AntiSpoofingSystem
anti_spoofing_system = AntiSpoofingSystem()

# Button dimensions and styles
button_width, button_height, font_size, font_size_total = 160, 40, 16, 20
corner_radius, border_color, border_width = 8, primary_color, 2

def create_button(parent, text, command):
    # Redesigned Button Style
    return ctk.CTkButton(parent, text=text, command=command, fg_color=primary_color,
                         text_color=secondary_color, hover_color=text_color, font=("Roboto", font_size),
                         width=button_width, height=button_height,
                         corner_radius=corner_radius, border_color=border_color, border_width=border_width)

# Directory for saving registered user images
save_directory = "Person"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

def register_interface():
    global buttonRegister, buttonCheckIn

    # Hide existing buttons
    buttonRegister.pack_forget()
    buttonCheckIn.pack_forget()
   

    # Call the run method of anti-spoofing_system to capture an image
    frame = anti_spoofing_system.run()

    if frame is not None:
        # Save the captured image for registration
        user_id = register_fill.get()
        image_filename = os.path.join(save_directory, f"{user_id}.png")
        cv2.imwrite(image_filename, frame)
        
        # Reset the image_captured flag for future registrations
        anti_spoofing_system.image_captured = False

        # Update UI to indicate successful registration
        status_label.configure(text=f"User {user_id} registered successfully.")

        # Reset UI elements as needed
        register_fill.delete(0, tk.END)  # Clear the entry field
        register_fill.pack_forget()
        buttonRegister.pack_forget()
        main_interface()  # Return to the main interface

def main_interface():
    global buttonRegister, buttonCheckIn, buttonBack

    buttonRegister = create_button(
        button_frame, "Register", register_interface)
    buttonRegister.pack(side="left", padx=10)

    buttonCheckIn = create_button(
        button_frame, "Check Database", check_in)
    buttonCheckIn.pack(side="left", padx=10)


def update_total_students_label():
    total_students_label.configure(
        text=f"Total Students: {len(checked_in_students)}")

def back():
    subprocess.Popen(["python", "app.py"])
    app.quit()

    ################### VIDEO AND ANTI-SPOOFING ###################
def update_frame():
    global none_spoofed_image

    # Call the run method of anti-spoofing_system
    frame = anti_spoofing_system.run()

    if frame is not None:
        # Convert the frame to PhotoImage
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, update_frame)  # Continue updating the frame

################### SUPERVISED MODEL ###################
# Face recognition model and user embeddings
RECOGNITION_THRESHOLD = 0.3
embedding_model = tf.keras.models.load_model('best_model.h5')
user_embeddings = {}

def preprocess_image(image):
    image = cv2.resize(image, (375, 375))  # Resize image to the expected input size
    image = tf.keras.applications.resnet50.preprocess_input(image)  # Adjust if using a different model
    return np.expand_dims(image, axis=0)


def generate_embedding(image):
    preprocessed_image = preprocess_image(image)
    return embedding_model.predict(preprocessed_image)[0]

def register_user(user_id):
    global status_label, buttonRegister, register_fill

    try:
        # Check if the anti-spoofing image has been captured
        if not anti_spoofing_system.image_captured:
            status_label.configure(text="Please complete the anti-spoofing.")
            return

        # Retrieve the captured image
        frame = anti_spoofing_system.get_captured_image()

        # Reset the image_captured flag for future registrations
        anti_spoofing_system.image_captured = False

        # Generate an embedding from the captured frame
        embedding = generate_embedding(frame)

        # Store the embedding with the user ID
        user_embeddings[user_id] = embedding

        # Update UI to indicate successful registration
        status_label.configure(text=f"User {user_id} registered successfully.")

        # Reset UI elements as needed
        register_fill.delete(0, tk.END)  # Clear the entry field
        register_fill.pack_forget()
        buttonRegister.pack_forget()
        main_interface()  # Return to the main interface

    except Exception as e:
        status_label.configure(text=f"Error during registration: {str(e)}")

def check_in():
    global status_label, checked_in_students

    try:
        frame = anti_spoofing_system.access_verified_image()

        if frame is None:
            status_label.configure(text="Please complete the anti-spoofing check first.")
            return

        new_embedding = generate_embedding(frame)
        min_distance = float('inf')
        recognized_user_id = "Unknown"

        # Load and compare with each user image in the Persons directory
        for filename in os.listdir(save_directory):
            if filename.endswith(".png"):
                user_id = filename.split('.')[0]
                user_image = cv2.imread(os.path.join(save_directory, filename))
                embedding = generate_embedding(user_image)
                distance = cosine(new_embedding, embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_user_id = user_id

        if min_distance <= RECOGNITION_THRESHOLD:
            checked_in_students.add(recognized_user_id)
            status_label.configure(text=f"Checked in: {recognized_user_id}")
        else:
            status_label.configure(text="User does not exist in Person folder.")

    except Exception as e:
        status_label.configure(text=f"Error during check-in: {str(e)}")

################### APP IMPLEMENTATION ###################
app = ctk.CTk()
app.title("Face Recognition Attendance System")
app.geometry("1440x810")
ctk.set_appearance_mode("light")

# Video and Anti-Spoofing Setup
cap = cv2.VideoCapture(0)
main_frame = ctk.CTkFrame(app, fg_color=secondary_color)
main_frame.pack(expand=True, fill='both')
video_label = tk.Label(main_frame, width=930, height=650)
video_label.grid(row=0, column=0, padx=10, columnspan=2)

# TOTAL STUDENT
checked_in_students = set()

total_students_label = ctk.CTkLabel(
    main_frame, text=f"Total Students: {len(checked_in_students)}", font=("Roboto", font_size_total))
total_students_label.grid(row=0, column=2, sticky="nw", padx=10)

verified_label = ctk.CTkLabel(
    main_frame, text="AntiSpoofed: False", font=("Roboto", font_size_total))
verified_label.grid(row=0, column=2, sticky="sw", padx=10)

# STATUS
status_label = ctk.CTkLabel(main_frame, text="", font=("Roboto", font_size), fg_color=secondary_color, text_color=text_color)
status_label.grid(row=1, column=0, padx=10, columnspan=2)

# BUTTON FRAME
button_frame = ctk.CTkFrame(main_frame, fg_color=secondary_color)
button_frame.grid(row=2, column=0, columnspan=3, pady=20)
main_interface()

################### LOOP ###################
update_frame()
app.mainloop()
cap.release()