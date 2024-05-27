import cv2
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime

print("Collecting Data...")

# Create Data directory if it doesn't exist
if not os.path.exists('Data'):
    os.makedirs('Data')

# Create or load the Excel file
excel_file = 'Data/faces_data.xlsx'
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=['ID', 'Name', 'Email', 'DateTime'])
    df.to_excel(excel_file, index=False)

# Load the existing data to find the next unique ID
df = pd.read_excel(excel_file)
next_id = len(df) + 1

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('/Users/wasifkarim/Desktop/Facial Recognition/Data/haarcascade_frontalface_default.xml')
faces_data = []
i = 0
name = input("Enter your name: ")
email = input("Enter your email ID: ")

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resize_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < 20 and i % 10 == 0:
            faces_data.append(resize_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= 20:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(20, -1)

if 'names.pkl' not in os.listdir('Data'):
    names = [name] * 20
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('Data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 20
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'emails.pkl' not in os.listdir('Data'):
    emails = [email] * 20
    with open('Data/emails.pkl', 'wb') as f:
        pickle.dump(emails, f)
else:
    with open('Data/emails.pkl', 'rb') as f:
        emails = pickle.load(f)
    emails = emails + [email] * 20
    with open('Data/emails.pkl', 'wb') as f:
        pickle.dump(emails, f)

if 'faces_data.pkl' not in os.listdir('Data'):
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('Data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

# Save the new entry to the Excel file
new_entry = pd.DataFrame([{
    'ID': next_id,
    'Name': name,
    'Email': email,
    'DateTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}])

df = pd.concat([df, new_entry], ignore_index=True)
df.to_excel(excel_file, index=False)

print(f"Data saved for {name} with ID {next_id}")