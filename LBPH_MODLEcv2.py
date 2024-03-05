import cv2
import os
import numpy as np

# Prepare training data
faces_folder = 'ImageAttendance'
face_images = []
face_labels = []
label_dict = {}  # Dictionary to map directory names to unique integer labels
label_counter = 0  # Counter to assign unique integer labels

# Traverse subdirectories skipping the root directory
for root, dirs, files in os.walk(faces_folder):
    if root != faces_folder:  # Skip the root directory
        label_dict[os.path.basename(root)] = label_counter
        label_counter += 1

# Collect training data
for root, dirs, files in os.walk(faces_folder):
    if root != faces_folder:  # Skip the root directory
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)
                face_labels.append(label_dict[label])  # Use the integer label from the dictionary
                face_images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer
recognizer.train(face_images, np.array(face_labels))

# Save the trained model to a file
recognizer.save('trained_model.yml')
