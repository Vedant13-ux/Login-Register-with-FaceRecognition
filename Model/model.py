import cv2
import numpy as np
import face_recognition
import os

#MongoDB Configiration
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
CONNECTION_STRING=os.getenv('MONGODB_URI')

#Get Database
def get_database():
    client = MongoClient(CONNECTION_STRING)
    return client['resoluteai']

# Get Previous Encodings
def get_prev_encodings():
    users=users_collection.find()
    users_list=[]
    encodings=[]
    for user in users:
        users_list.append(user['username'])
        encodings.append(user['encoding'])
    return encodings, users_list

def insert_user(encoding, username):
    try:
        encoding=encoding.tolist()
        users_collection.insert_one({"encoding":encoding, "username":username})
        prev_encodings.append(encoding)
        users.append(username)

    except Exception as e:
        print(e)

def user_encodings():
    return prev_encodings, users

# Defining Database Variables
database=get_database()
users_collection=database['users']
prev_encodings, users= get_prev_encodings()

# # Recognition
# cap = cv2.VideoCapture(0)
# while True:
#     _, img = cap.read()
#     imgS = cv2.resize(img,(0,0),None,0.25,0.25) #Resize the image to Factor of 0.25
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
#     # Encoding Current Image
#     current_face = face_recognition.face_locations(imgS)
#     current_encoded_face = face_recognition.face_encodings(imgS,current_face)
    
#     if len(current_face)==1:
#         matches = face_recognition.compare_faces(prev_encodings, current_encoded_face[0])
#         faceDis = face_recognition.face_distance(prev_encodings, current_encoded_face[0])
#         print(matches,faceDis)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             y1,x2,y2,x1 = current_face[0]
#             y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,users[matchIndex],(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#             print('Identified')
#         else:
#             y1,x2,y2,x1 = current_face[0]
#             y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,"Unknown",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#             print('New Face Identified')
#             insert_user(users_collection,current_encoded_face[0])
 
    
#     elif len(current_face)==0:
#         print("No Face Detected")
    
#     else:
#         print("Many Faces Detected, Try Again")

#     #Displaying the Frame      
#     cv2.imshow('Webcam',img)
#     cv2.waitKey(1)