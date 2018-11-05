
#getting libraries
import cv2
import requests
import time

#setting the API parameters
TOKEN = "A1E-yaJ6gGCa5iJkNhX1Rpkvl1MzsvtZFj"
#setting the device label
DEVICE_LABEL = "test"

#creating the function to create the recognition-payload
def build_payload(variable_1, value_1, variable_2, value_2):
    payload = {variable_1: value_1,
               variable_2: value_2}
    return payload

#creating the posting API function
def post_request(payload):
    # Creates the headers for the HTTP requests
    url = "http://things.ubidots.com"
    url = "{}/api/v1.6/devices/{}".format(url, DEVICE_LABEL)
    headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}

    # Makes the HTTP requests
    status = 400
    attempts = 0
    while status >= 400 and attempts <= 5:
        req = requests.post(url=url, headers=headers, json=payload)
        status = req.status_code
        attempts += 1
        time.sleep(1)

    # Processes results
    if status >= 400:
        print("[ERROR] Could not send data after 5 attempts, please check \
            your token credentials and internet connection")
        return False               

    print("[INFO] request made properly, your device is updated")
    return True

#import numpy as np
#import os

##Recognizer definitions
#define the face recognizer API from the opencv library
recognizer = cv2.face.LBPHFaceRecognizer_create(neighbors = 10)
#read the pre--trainned model
recognizer.read('trainer/trainer_default.yml')
#set the identity classifier and his type
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
#some other examples to be used for recognition
#neighbors = 20, grid_x = 20, grid_y = 20
#recognizer  cv2.face.FisherFaceRecognizer_create(num_components = 100)
#recognizer  cv2.face.EigenFaceRecognizer_create()

##Apearence definitions
#define the font for the ilustration on camera, if it is necessary
font = cv2.FONT_HERSHEY_SIMPLEX
#set the names, respectivelly with the trainning data code
names = ['Marcelo','Renan','Kato']

#iniciate id counter
id = 0

##Set the camera definitions
#initialize and start realtime video capture
#cam = cv2.VideoCapture(0)
#or you can use remote camera 
cam = cv2.VideoCapture('http://10.33.110.2:8080/video')
#set the video configurations
cam.set(3, 640) #set video widht
cam.set(4, 480) #set video height
#define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

#run forever
while True:
    #get the image from the camera
    ret, img =cam.read()
    
    ##Prepare the data
    #you can flip, if the raspberry camera is upside-down, it is commom!!
    #img = cv2.flip(img, 0) # Flip vertically
    #set the image to black&&white, since we are interested only on shapes
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray = img
    
    #define the face detector, to detect different scalled images 
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    
    for(x,y,w,h) in faces:
        #preprare the rectangle to show on the video
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        #receives the cofidence and the id of the respective face
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        #check if confidence is less them 100, since "0" is perfect match 
        if (confidence < 100):
            #define the name respective with the most suitable face
            id = names[id]
            #show the confidence level on that particular face
            confidence = "  {0}%".format(round(100 - confidence))
            #send the confidence level to ubidots
            person = str(id).lower()
            payload = build_payload("confidence",confidence,person,1)
            time.sleep(2)
        else:
            #if the match is poorly, show the unkonw marker
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            time.sleep(2)

        #print the confidence and the name on the image
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    #show the captured image and treated image
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff #press 'ESC' for exiting video
    if k == 27:
        break
#do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
