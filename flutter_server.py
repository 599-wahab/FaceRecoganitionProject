from flask import Flask, Response
import cv2
import face_recognition
import threading
import winsound
import pickle
import os
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

frequency = 2300  # Set the frequency in Hertz
duration = 1300  # Set the duration in milliseconds

# Global variables for face recognition
faces_encodings = []
labels = []

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def prepare_training_data(data_folder):
    global faces_encodings, labels
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

# Check if trained model exists
trained_model_file = 'trained_model.pkl'
if os.path.isfile(trained_model_file):
    # Load the trained model
    with open(trained_model_file, 'rb') as file:
        faces_encodings, labels = pickle.load(file)
else:
    # Prepare training data and train the model
    data_folder = 'FaceRecognitionImgs'
    prepare_training_data(data_folder)
    # Save the trained model
    with open(trained_model_file, 'wb') as file:
        pickle.dump((faces_encodings, labels), file)

def process_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Cannot read frame from camera.")
            break

        # Convert images to RGB format for face recognition
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in each frame
        face_locations = face_recognition.face_locations(rgb_img, model='hog')  # Using HOG model
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        # Draw boxes around faces and display names
        draw_boxes(frame, face_encodings, face_locations)

        # Send the results back to Flutter app
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def draw_boxes(img, encodings, locations):
    global faces_encodings, labels
    for encodeFace, faceLoc in zip(encodings, locations):
        matches = face_recognition.compare_faces(faces_encodings, encodeFace, tolerance=0.5)  # Adjust tolerance
        faceDis = face_recognition.face_distance(faces_encodings, encodeFace)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = labels[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (128, 0, 128), 1)
                accuracy = 1 - faceDis[matchIndex]
                print(f"Person: {name}, Accuracy: {accuracy}")
            else:
                name = "Unknown"
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1)
                accuracy = 1 - faceDis[matchIndex]
                print(f"Person: {name}, Accuracy: {accuracy}")
                winsound.Beep(frequency, duration)

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=process_frames).start()
    app.run(debug=True)
