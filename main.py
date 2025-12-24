import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to training images
path = 'Students_images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to find encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to mark attendance
def markAttendance(name, status="Recognized"):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        # Only write a new attendance entry if this name hasn't been recorded yet
        # (prevents repeating the same entry every frame). External logic
        # ensures markAttendance is invoked once per appearance.
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime('%Y-%m-%d')  # Current date
            timeString = now.strftime('%H:%M:%S')  # Current time
            f.writelines(f'\n{name},{status},{dateString},{timeString}')

# Encode known images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Keep track of labels we've already reported for the current presence
# and clear them after no faces are seen for a short interval.
reported_labels = set()
absence_frames = 0
ABSENCE_THRESHOLD = 30  # number of frames with no faces before reset

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # If any faces are found, reset absence counter and process each face.
    if len(encodesCurFrame) > 0:
        absence_frames = 0
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            # Determine label
            if faceDis[matchIndex] < 0.5:  # Adjusted threshold
                label = classNames[matchIndex].upper()
                status = "Recognized"
            else:
                label = "Unknown"
                status = "Unrecognized"

            # Only report this label once while that person is present
            if label not in reported_labels:
                # Scale back face location to original frame for drawing
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                if status == "Recognized":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Unrecognized", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Log attendance once per appearance
                markAttendance(label, status)
                reported_labels.add(label)

            else:
                # Even if we've already reported, still draw the box and label
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                if faceDis[matchIndex] < 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, classNames[matchIndex].upper(), (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Unrecognized", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    else:
        # No faces found in this frame; increment absence counter and
        # clear reported labels after a threshold so new people can be reported.
        absence_frames += 1
        if absence_frames > ABSENCE_THRESHOLD:
            reported_labels.clear()

    # Display webcam feed
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
