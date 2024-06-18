import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import datetime
import customtkinter as ctk
import os
import pickle
import face_recognition
import time
import winsound  # For sound feedback on Windows

FPS = 23  # Frames per second
trained_model_file = 'trained_model.pkl'

# Sound parameters
known_frequency = 2300  # Frequency for known faces in Hertz
known_duration = 130  # Duration for known faces in milliseconds
unknown_frequency = 3500  # Frequency for unknown faces in Hertz
unknown_duration = 130  # Duration for unknown faces in milliseconds

# Function to load images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

# Function to prepare training data and generate encodings
def prepare_training_data(data_folder):
    labels = []
    faces_encodings = []

    for person_name in os.listdir(data_folder):
        person_folder = os.path.join(data_folder, person_name)
        if os.path.isdir(person_folder):
            images = load_images_from_folder(person_folder)
            for image in images:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                if face_locations:
                    encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    if encodings:
                        faces_encodings.append(encodings[0])
                        labels.append(person_name)
    return faces_encodings, labels

def train_model():
    data_folder = 'FaceRecognitionImgs'
    faces_encodings, labels = prepare_training_data(data_folder)
    with open(trained_model_file, 'wb') as file:
        pickle.dump((faces_encodings, labels), file)

# Check if trained model file exists, if not, create it
if not os.path.isfile(trained_model_file):
    train_model()

class WebcamWidget:
    def __init__(self, master=None, *args, **kwargs):
        # Initialize cameras
        self.vid1 = cv2.VideoCapture(0)  # First camera
        self.vid2 = cv2.VideoCapture(1)  # Second camera

        self.frame = ctk.CTkFrame(master)
        self.frame.pack(expand=True, fill="both")

        # Labels for displaying video from two cameras
        self.label1 = ctk.CTkLabel(self.frame)
        self.label1.grid(row=0, column=0, padx=5, pady=5)

        self.label2 = ctk.CTkLabel(self.frame)
        self.label2.grid(row=0, column=1, padx=5, pady=5)

        # Buttons for user interaction
        self.register_button = ctk.CTkButton(self.frame, text="Register", command=self.register_user)
        self.register_button.grid(row=1, column=0, padx=5, pady=5)
        self.show_record_button = ctk.CTkButton(self.frame, text="Show Records", command=self.show_records)
        self.show_record_button.grid(row=1, column=1, padx=5, pady=5)
        self.search_button = ctk.CTkButton(self.frame, text="Search", command=self.search)
        self.search_button.grid(row=1, column=2, padx=5, pady=5)

        self.auth_status_label = ctk.CTkLabel(self.frame, text="Status: N/A")
        self.auth_status_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        self.update()

    def update(self):
        any_known_face = False

        if self.vid1.isOpened():
            ret1, frame1 = self.vid1.read()
            if ret1:
                rgb_im1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                any_known_face = self.process_frame(rgb_im1, self.label1) or any_known_face

        if self.vid2.isOpened():
            ret2, frame2 = self.vid2.read()
            if ret2:
                rgb_im2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                any_known_face = self.process_frame(rgb_im2, self.label2) or any_known_face

        # Update the status label
        if any_known_face:
            self.auth_status_label.configure(text="Status: Authorized")
        else:
            self.auth_status_label.configure(text="Status: No Face :)")

        self.auth_status_label.after(int(1000 / FPS), self.update)

    def process_frame(self, frame, label):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        any_known_face = False

        with open(trained_model_file, 'rb') as file:
            faces_encodings, labels = pickle.load(file)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown

            if True in matches:
                first_match_index = matches.index(True)
                name = labels[first_match_index]
                color = (0, 255, 0)  # Green for known
                any_known_face = True
                # Play sound for known face
                winsound.Beep(known_frequency, known_duration)
            else:
                # Play sound for unknown face
                winsound.Beep(unknown_frequency, unknown_duration)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(processed_frame)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        return any_known_face

    def register_user(self):
        registration_window = ctk.CTkToplevel(self.frame)
        registration_window.title("Register User")
        registration_window.geometry("400x300")

        name_label = ctk.CTkLabel(registration_window, text="Enter Name:")
        name_label.pack(pady=10)
        name_entry = ctk.CTkEntry(registration_window)
        name_entry.pack(pady=10)

        capture_button = ctk.CTkButton(registration_window, text="Capture Images", command=lambda: self.capture_images(name_entry.get(), registration_window))
        capture_button.pack(pady=10)

    def capture_images(self, name, registration_window):
        folder_path = os.path.join('FaceRecognitionImgs', name)
        os.makedirs(folder_path, exist_ok=True)

        for i in range(3):
            if self.vid1.isOpened():
                ret, frame = self.vid1.read()
                if ret:
                    image_path = os.path.join(folder_path, f'{name}_{i + 1}.jpg')
                    cv2.imwrite(image_path, frame)
                    time.sleep(3)

        # Update training data
        self.update_training_model()

        registration_window.destroy()

    def update_training_model(self):
        train_model()

    def show_records(self):
        print("Show records functionality (to be implemented).")

    def search(self):
        print("Search functionality (to be implemented).")

class LoginPage:
    def __init__(self, master, on_login_success):
        self.master = master
        self.on_login_success = on_login_success
        self.master.title("Login")
        self.master.geometry("800x600")

        self.frame = ctk.CTkFrame(master)
        self.frame.pack(expand=True, fill="both")
        self.title_label = ctk.CTkLabel(self.frame, text="SLCV", font=("Arial", 24))
        self.title_label.pack(pady=10)

        self.username_label = ctk.CTkLabel(self.frame, text="Username")
        self.username_label.pack(pady=5)
        self.username_entry = ctk.CTkEntry(self.frame)
        self.username_entry.pack(pady=5)

        self.password_label = ctk.CTkLabel(self.frame, text="Password")
        self.password_label.pack(pady=5)
        self.password_entry = ctk.CTkEntry(self.frame, show="*")
        self.password_entry.pack(pady=5)

        self.login_button = ctk.CTkButton(self.frame, text="Login", command=self.check_login)
        self.login_button.pack(pady=10)

    def check_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if username == "admin" and password == "admin":
            self.on_login_success()
        

def open_webcam_widget():
    root = ctk.CTk()
    root.title("Webcam Face Recognition")
    root.geometry("800x600")
    WebcamWidget(root)
    root.mainloop()

def on_login_success():
    login_window.destroy()
    open_webcam_widget()

if __name__ == "__main__":
    login_window = ctk.CTk()
    LoginPage(login_window, on_login_success)
    login_window.mainloop()
