import cv2
import numpy as np
import os 
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

from win32com.client import Dispatch

def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)


video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('data/names.pkl','rb') as w:
    LABELS=pickle.load(w)

with open('data/face_data.pkl','rb') as f:
    FACES=pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)

imgbackground=cv2.imread("bg.png")#picture to be inputted

COL_NAMES=['NAME','TIME']

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=facedetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in face:
        crop_img=frame[y:y+h,x:x+w,:]
        resized_img=cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist=os.path.isfile("attendance/attendance_"+date+".csv")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame,(x,y),(x+w,y+h),(50,50,255),1)
        attendance=[str(output[0]),str(timestamp)]
    imgbackground[162:162+480,55:55+640]=frame

    
    cv2.imshow("frame",imgbackground)
    k=cv2.waitKey(1)
    if k==ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)

        if exist:
            with open("attendance/attendance_"+date+".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()

        else:
            with open("attendance/attendance_"+date+".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()

    if k==ord('q'):
        break

video.release() 
cv2.destroyAllWindows()