import cv2
import numpy as np
import face_recognition
import os
import time
import threading
import winsound
import pickle

frequency = 2300  # Set the frequency in Hertz
duration = 1300  # Set the duration in milliseconds

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

# Check if trained model exists
trained_model_file = 'trained_model.pkl'
if os.path.isfile(trained_model_file):
    # Load the trained model
    with open(trained_model_file, 'rb') as file:
        faces_encodings, labels = pickle.load(file)
else:
    # Prepare training data and train the model
    data_folder = 'FaceRecognitionImgs'
    faces_encodings, labels = prepare_training_data(data_folder)
    # Save the trained model
    with open(trained_model_file, 'wb') as file:
        pickle.dump((faces_encodings, labels), file)

def process_frames():
    # Initialize capture devices within the thread
    cap = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(4)
    cap2 = cv2.VideoCapture(1)
    cap3 = cv2.VideoCapture(2)

    while True:
        start_time = time.time()

        ret, img = cap.read()
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        ret3, img3 = cap3.read()

        if not (ret and ret1 and ret2 and ret3):
            print("Error: Cannot read frames from cameras.")
            break

        # Convert images to RGB format for face recognition
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        rgb_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in each frame
        frames = [(rgb_img, img), (rgb_img1, img1), (rgb_img2, img2), (rgb_img3, img3)]
        for rgb_frame, frame in frames:
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')  # Using HOG model
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Draw boxes around faces and display names
            draw_boxes(frame, face_encodings, face_locations)

        # Combine the four images into one grid
        top_row = np.hstack((img, img1))
        bottom_row = np.hstack((img2, img3))
        combined_image = np.vstack((top_row, bottom_row))

        # Display the combined image
        cv2.imshow('Webcams', combined_image)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_processing_time = time.time() - start_time

    # Release the capture devices within the thread
    cap.release()
    cap1.release()
    cap2.release()
    cap3.release()

def draw_boxes(img, encodings, locations):
    for encodeFace, faceLoc in zip(encodings, locations):
        matches = face_recognition.compare_faces(faces_encodings, encodeFace, tolerance=0.5)  # Adjust tolerance
        faceDis = face_recognition.face_distance(faces_encodings, encodeFace)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = labels[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), (128, 0, 128), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 128), 2)
                accuracy = 1 - faceDis[matchIndex]
                print(f"Person: {name}, Accuracy: {accuracy}")
            else:
                name = "Unknown"
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                accuracy = 1 - faceDis[matchIndex]
                print(f"Person: {name}, Accuracy: {accuracy}")
                winsound.Beep(frequency, duration)

# Start a separate thread for processing frames
thread = threading.Thread(target=process_frames)
thread.start()

# Wait for the thread to finish
thread.join()

# Release the capture devices
cv2.destroyAllWindows()
