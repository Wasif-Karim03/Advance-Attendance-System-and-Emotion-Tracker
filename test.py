from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os

print("Collecting Data...")

# Create Data directory if it doesn't exist
if not os.path.exists('Data'):
    os.makedirs('Data')

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('/Users/wasifkarim/Desktop/Facial Recognition/Data/haarcascade_frontalface_default.xml')

with open('Data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('Data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

with open('Data/emails.pkl', 'rb') as f:
    EMAILS = pickle.load(f)

# Correctly instantiate the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resize_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resize_img)
        
        # Get the index of the recognized label
        label_index = LABELS.index(output[0])
        email = EMAILS[label_index]
        
        # Draw green rectangles around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv2.rectangle(frame, (x, y-60), (x+w, y), (255, 0, 0), -1)
        
        # Display the output in white
        cv2.putText(frame, str(output[0]), (x, y-45), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # White color
        cv2.putText(frame, email, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # White color
    
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()