import cv2
import numpy as np
import face_recognition
imgAb = face_recognition.load_image_file('ImagesBasic/e.jpg')
imgAb = cv2.cvtColor(imgAb, cv2.COLOR_BGR2RGB)
imageTest = face_recognition.load_image_file('ImagesBasic/Etest.jpg')
imgTest = cv2.cvtColor(imageTest, cv2.COLOR_BGR2RGB)
# Find face locations and encodings
facloc = face_recognition.face_locations(imgAb)
encodab = face_recognition.face_encodings(imgAb)
facloctest = face_recognition.face_locations(imgTest)
encodabtest = face_recognition.face_encodings(imgTest)
# Convert lists to NumPy arrays
encodab = np.array(encodab)
encodabtest = np.array(encodabtest)
# Check if any faces were found
if len(facloc) > 0 and len(facloctest) > 0:
    # Compare face encodings
    distances = face_recognition.face_distance(encodab, encodabtest)
    results = [distance <= 0.5 for distance in distances]
    # Draw rectangles around faces and annotate with "Match" or "Not a match"
    for i, loc in enumerate(facloc):
        cv2.rectangle(imgAb, (loc[3], loc[0]), (loc[1], loc[2]), (255, 0, 255), 2)
        if results[i]:
            cv2.putText(imgAb, "Match", (loc[1], loc[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(imgAb, "Not a match", (loc[1], loc[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for i, loc in enumerate(facloctest):
        cv2.rectangle(imgTest, (loc[3], loc[0]), (loc[1], loc[2]), (46, 185, 136, 96), 2)
        if results[i]:
            cv2.putText(imgTest, "Match", (loc[1], loc[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(imgTest, "Not a match", (loc[1], loc[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('ab', imgAb)
    cv2.imshow('ab test', imgTest)
    cv2.waitKey(0)
else:
    print("No faces found in one of the images.")


    