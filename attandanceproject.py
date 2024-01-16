import cv2
import numpy as np
import face_recognition
import os
import time

path = 'ImageAttendance'
imgs = []
classNames = []
myList = os.listdir(path)
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
        if encod is not None:
            encodeList.append(encod[0])
    return encodeList

encodeListKnow = findEncodings(imgs)
print(len(encodeListKnow))

cap = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    success, img = cap.read()
    success1, img1 = cap1.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    imgS1 = cv2.resize(img1, (0, 0), None, 0.25, 0.25)
    imgS1 = cv2.cvtColor(imgS1, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Find all face locations and encodings in the second camera frame
    faceCurFrame1 = face_recognition.face_locations(imgS1)
    encodeCurFrame1 = face_recognition.face_encodings(imgS1, faceCurFrame1)

    # Draw boxes around faces in the first camera frame and display names
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print(faceDis)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)

                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1 * 4, y1 * 4), (x2 * 4, y2 * 4), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                name = "Unknown"
                print(name)

    # Draw boxes around faces in the second camera frame and display names
    for encodeFace, faceLoc in zip(encodeCurFrame1, faceCurFrame1):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print(faceDis)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)

                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img1, (x1 * 4, y1 * 4), (x2 * 4, y2 * 4), (0, 255, 0), 2)
                cv2.rectangle(img1, (x1 + 6,  y2 - 6), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img1, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                name = "Unknown"
                print(name)

    frame_processing_time = time.time() - start_time
    cv2.imshow('webcam', img)
    cv2.imshow('webcam1', img1)

    if cv2.waitKey(1) == ord('q'):
        break
