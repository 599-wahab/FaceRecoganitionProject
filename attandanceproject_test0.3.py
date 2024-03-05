import cv2
import numpy as np
import face_recognition
import os
import time
import threading
import winsound

path = 'ImageAttendance'
imgs = []
classNames = []
myList = os.listdir(path)

frequency = 2300  # Set the frequency in Hertz
duration = 1300  # Set the duration in milliseconds

print(myList)

for c1 in myList:
    curImg = cv2.imread(f'{path}/{c1}')
    imgs.append(curImg)
    classNames.append(os.path.splitext(c1)[0])

print(classNames)

def findEncodings(imgs):
    encodeList = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encod = face_recognition.face_encodings(img)
        if encod:
            encodeList.append(encod[0])
    return encodeList

encodeListKnow = findEncodings(imgs)
print(len(encodeListKnow))

# Initialize capture devices within the process_frames function
cap = None
cap1 = None
cap2 = None
cap3 = None

# Frame Skipping
skip_frames = 5
frame_count = 0

def process_frames():
    global frame_count, cap, cap1, cap2, cap3
    # Initialize capture devices within the thread
    cap = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(4)
    cap2 = cv2.VideoCapture(1)
    cap3 = cv2.VideoCapture(2)

    # Get the frame dimensions for dynamically adjusting the screen size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        start_time = time.time()

        success, img = cap.read()
        success1, img1 = cap1.read()
        success2, img2 = cap2.read()
        success3, img3 = cap3.read()

        imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)  # Increase resolution
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        imgS1 = cv2.resize(img1, (0, 0), None, 0.5, 0.5)  # Increase resolution
        imgS1 = cv2.cvtColor(imgS1, cv2.COLOR_BGR2RGB)

        imgS2 = cv2.resize(img2, (0, 0), None, 0.5, 0.5)  # Increase resolution
        imgS2 = cv2.cvtColor(imgS2, cv2.COLOR_BGR2RGB)

        imgS3 = cv2.resize(img3, (0, 0), None, 0.5, 0.5)  # Increase resolution
        imgS3 = cv2.cvtColor(imgS3, cv2.COLOR_BGR2RGB)

        if frame_count % skip_frames == 0:
            # Find all face locations and encodings in the current frame
            faceCurFrame = face_recognition.face_locations(imgS, model='hog')  # Using HOG model
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            # Find all face locations and encodings in the second camera frame
            faceCurFrame1 = face_recognition.face_locations(imgS1, model='hog')  # Using HOG model
            encodeCurFrame1 = face_recognition.face_encodings(imgS1, faceCurFrame1)

            # Find all face locations and encodings in the third camera frame
            faceCurFrame2 = face_recognition.face_locations(imgS2, model='hog')  # Using HOG model
            encodeCurFrame2 = face_recognition.face_encodings(imgS2, faceCurFrame2)

            # Find all face locations and encodings in the fourth camera frame
            faceCurFrame3 = face_recognition.face_locations(imgS3, model='hog')  # Using HOG model
            encodeCurFrame3 = face_recognition.face_encodings(imgS3, faceCurFrame3)

            # Draw boxes around faces in the first camera frame and display names
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnow, encodeFace, tolerance=0.5)  # Adjust tolerance
                faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)

                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()

                        y1, x2, y2, x1 = faceLoc
                        cv2.rectangle(img, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (128, 0, 128), 2)
                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 128), 2)
                        accuracy = 1 - faceDis[matchIndex]
                        print(f"Person: {name}, Accuracy: {accuracy}")

                        # winsound.Beep(frequency, duration)

                    else:
                        name = "Unknown"
                        y1, x2, y2, x1 = faceLoc
                        cv2.rectangle(img, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (0, 0, 255), 2)  # Red box
                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        accuracy = 1 - faceDis[matchIndex]
                        print(f"Person: {name}, Accuracy: {accuracy}")

                        winsound.Beep(frequency, duration)

            # Draw boxes around faces in the second camera frame and display names
            for encodeFace, faceLoc in zip(encodeCurFrame1, faceCurFrame1):
                matches = face_recognition.compare_faces(encodeListKnow, encodeFace, tolerance=0.5)  # Adjust tolerance
                faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)

                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()

                        y1, x2, y2, x1 = faceLoc
                        cv2.rectangle(img1, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (128, 0, 128), 2)
                        cv2.putText(img1, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 128), 2)
                        accuracy = 1 - faceDis[matchIndex]
                        print(f"Person: {name}, Accuracy: {accuracy}")

                        # winsound.Beep(frequency, duration)

                    else:
                        name = "Unknown"
                        y1, x2, y2, x1 = faceLoc
                        cv2.rectangle(img1, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (0, 0, 255), 2)  # Red box
                        cv2.putText(img1, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        accuracy = 1 - faceDis[matchIndex]
                        print(f"Person: {name}, Accuracy: {accuracy}")

                        winsound.Beep(frequency, duration)

            # Draw boxes around faces in the third camera frame and display names
            for encodeFace, faceLoc in zip(encodeCurFrame2, faceCurFrame2):
                matches = face_recognition.compare_faces(encodeListKnow, encodeFace, tolerance=0.5)  # Adjust tolerance
                faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)

                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()

                        y1, x2, y2, x1 = faceLoc
                        cv2.rectangle(img2, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (128, 0, 128), 2)
                        cv2.putText(img2, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 128), 2)
                        accuracy = 1 - faceDis[matchIndex]
                        print(f"Person: {name}, Accuracy: {accuracy}")

                        # winsound.Beep(frequency, duration)

                    else:
                        name = "Unknown"
                        y1, x2, y2, x1 = faceLoc
                        cv2.rectangle(img2, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (0, 0, 255), 2)  # Red box
                        cv2.putText(img2, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        accuracy = 1 - faceDis[matchIndex]
                        print(f"Person: {name}, Accuracy: {accuracy}")

                        winsound.Beep(frequency, duration)

            # Draw boxes around faces in the fourth camera frame and display names
            for encodeFace, faceLoc in zip(encodeCurFrame3, faceCurFrame3):
                matches = face_recognition.compare_faces(encodeListKnow, encodeFace, tolerance=0.5)  # Adjust tolerance
                faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)

                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()

                        y1, x2, y2, x1 = faceLoc
                        cv2.rectangle(img3, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (128, 0, 128), 2)
                        cv2.putText(img3, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 0, 128), 2)
                        accuracy = 1 - faceDis[matchIndex]
                        print(f"Person: {name}, Accuracy: {accuracy}")

                        # winsound.Beep(frequency, duration)

                    else:
                        name = "Unknown"
                        y1, x2, y2, x1 = faceLoc
                        cv2.rectangle(img3, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (0, 0, 255), 2)  # Red box
                        cv2.putText(img3, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        accuracy = 1 - faceDis[matchIndex]
                        print(f"Person: {name}, Accuracy: {accuracy}")

                        winsound.Beep(frequency, duration)

            frame_processing_time = time.time() - start_time

            # Combine the four images into one grid
            top_row = np.hstack((img, img1))
            bottom_row = np.hstack((img2, img3))
            combined_image = np.vstack((top_row, bottom_row))

            # Dynamically adjust the window size based on the frame dimensions
            cv2.namedWindow('Webcams', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Webcams', width * 2, height * 2)  # Multiply by 2 for higher resolution
            cv2.imshow('Webcams', combined_image)

            if cv2.waitKey(1) == ord('q'):
                break

        frame_count += 1

    # Release the capture devices within the thread
    cap.release()
    cap1.release()
    cap2.release()
    cap3.release()

# Start a separate thread for processing frames
thread = threading.Thread(target=process_frames)
thread.start()

# Wait for the thread to finish
thread.join()

# Release the capture devices
cv2.destroyAllWindows()
