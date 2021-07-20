from logging import debug
from flask import Flask, views, request, render_template, url_for, Response, session
from flask_session import Session
from flask.globals import session
from numpy.lib import imag
import os 
from Model import model
from camera import Camera
import cv2
import face_recognition
import numpy as np

prev_encodings, users=model.user_encodings()
# username=""

# Initializing the Flask Application 
app=Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
sess = Session()

# Regsiter route
@app.route('/register', methods=["GET","POST"])
def register():
    return render_template('index.html')

# Login route
@app.route('/login', methods=["GET","POST"])
def login():
    return render_template('login.html')

# Image Capturing Route
@app.route('/capture', methods=["GET","POST"])
def video_stream():
    if request.method=="POST":
        username=request.form['username']
        session['username']=username
        return render_template('video_capture.html', username=username)

################# Register#############
def gen(username):
   # Recognition
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25) #Resize the image to Factor of 0.25
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        # Encoding Current Image
        current_face = face_recognition.face_locations(imgS)
        current_encoded_face = face_recognition.face_encodings(imgS,current_face)
        
        if len(current_face)==1:
            matches = face_recognition.compare_faces(prev_encodings, current_encoded_face[0])
            faceDis = face_recognition.face_distance(prev_encodings, current_encoded_face[0])
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1,x2,y2,x1 = current_face[0]
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,users[matchIndex],(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                print('Identified')
            else:
                y1,x2,y2,x1 = current_face[0]
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,"Identifying",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                print('New Face Identified')
                if username is None:
                    cv2.putText(img,"Unknown",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                else:
                    model.insert_user(current_encoded_face[0], username)
    
        
        elif len(current_face)==0:
            cv2.putText(img,"No Face Detected",(160,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )
            print("No Face Detected")
        
        else:
            for face in current_face:
                y1,x2,y2,x1 = face
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,"Many Faces Detected",(150,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )
            print("Many Faces Detected, Try Again")

        #Displaying the Frame    
        _, buffer=cv2.imencode('.jpg', img)  
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(session['username']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


#############Login#############
def gen_login():
    # Recognition
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25) #Resize the image to Factor of 0.25
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        # Encoding Current Image
        current_face = face_recognition.face_locations(imgS)
        current_encoded_face = face_recognition.face_encodings(imgS,current_face)
        
        if len(current_face)==1:
            matches = face_recognition.compare_faces(prev_encodings, current_encoded_face[0])
            faceDis = face_recognition.face_distance(prev_encodings, current_encoded_face[0])
            print(matches,faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1,x2,y2,x1 = current_face[0]
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,users[matchIndex],(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                print('Identified')
            else:
                y1,x2,y2,x1 = current_face[0]
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,"Unknown",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                print('New Face Identified')
    
        
        elif len(current_face)==0:
            cv2.putText(img,"No Face Detected",(160,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )
            print("No Face Detected")
        
        else:
            for face in current_face:
                y1,x2,y2,x1 = face
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,"Many Faces Detected",(150,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )
            print("Many Faces Detected, Try Again")

        #Displaying the Frame    
        _, buffer=cv2.imencode('.jpg', img)  
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed_login')
def video_feed_login():
    return Response(gen_login(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

            

if __name__=="__main__":
    sess.init_app(app)
    app.debug=True
    app.run()