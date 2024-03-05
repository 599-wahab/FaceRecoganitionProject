import os
import cv2
import numpy as np
import face_recognition

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

data_folder = 'ImageAttendance'
faces_encodings, labels = prepare_training_data(data_folder)

# Now you have faces_encodings and labels to use for training a classifier, like SVM or KNN.
# You can use faces_encodings as your X (features) and labels as your y (target).

# For example, you can use scikit-learn to train a classifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(faces_encodings, labels, test_size=0.2, random_state=42)

# Initialize and train SVM classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Evaluate classifier
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)
