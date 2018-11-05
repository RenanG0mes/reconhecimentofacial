#getting libraries
import cv2
import requests
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO

##GPIOs setup
GPIO.setmode(GPIO.BOARD)

#set pwm output
GPIO.setup(32,GPIO.OUT)
p = GPIO.PWM(32,50)
p.start(7.5)

#set digital outputs
GPIO.setup(36,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(38,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(40,GPIO.OUT,initial=GPIO.LOW)

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

#creating the get API function
def get_var(device, variable):
    try:
        url = "http://things.ubidots.com/"
        url = url + \
            "api/v1.6/devices/{0}/{1}/".format(device, variable)
        headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}
        req = requests.get(url=url, headers=headers)
        return req.json()['last_value']['value']
    except:
        pass

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
#cam = cv2.VideoCapture('http://10.33.110.2:8080/video')

#or uses the raspberry camera
cam = PiCamera()
#set the video configurations
cam.resolution = (640, 480)
cam.framerate = 32
rawCapture = PiRGBArray(cam, size=(640,480))

#give a time to initalize the camera
time.sleep(0.1)

#cam.set(3, 640) #set video widht
#cam.set(4, 480) #set video height


#define min window size to be recognized as a face
minW = 64 #min width
minH = 48 #min height

#run forever
for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):    

    #get the image from the camera
    #ret, img = cam.read()
    img = frame.array
    
    ##Prepare the data
    #you can flip, if the raspberry camera is upside-down, it is commom!!
    #img = cv2.flip(img, 0) # Flip vertically

    #set the image to black&&white, since we are interested only on shapes
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

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
        #get the unlockable variable
        unlock = get_var(DEVICE_LABEL, 'control')

        
        #check if confidence is less them 100, since "0" is perfect match
        #and if the door is unlockable
        if (confidence < 80 and unlock == 0.0):
            
            #define the name respective with the most suitable face
            id = names[id]
            #show the confidence level on that particular face
            confid_show = "  {0}%".format(round(100 - confidence))
            #get the format to sent to the ubidots
            confid_send = 100-confidence
            #format the name, since the label is lowercase
            person = str(id).lower()
            
            #open the gate
            GPIO.output(36,GPIO.LOW)
            GPIO.output(38,GPIO.HIGH)
            GPIO.output(40,GPIO.LOW)
            
            #send the confidence level to ubidots
            payload = build_payload("confidence",confid_send,person,1)
            post_request(payload)
            p.ChangeDutyCycle(2.5)
            #time.sleep(2)
            
        elif unlock == 1.0:
            
            #close the gate
            GPIO.output(38,GPIO.LOW)
            GPIO.output(36,GPIO.HIGH)
            GPIO.output(40,GPIO.HIGH)
            pwm_value = get_var(DEVICE_LABEL, 'slider')
            pwm_c = pwm_value/10.0 + 2.5
            p.ChangeDutyCycle(pwm_c)
            
            #if the match is poorly, show the unkonw marker
            id = "unknown"
            confid_show = "  {0}%".format(round(100 - confidence))
            #get the format to sent to the ubidots
            confid_send = 100-confidence
        else:
            #if the match is poorly, show the unkonw marker
            id = "unknown"
            confid_show = "  {0}%".format(round(100 - confidence))
            #get the format to sent to the ubidots
            confid_send = 100-confidence
            
            #close the gate
            GPIO.output(36,GPIO.LOW)
            GPIO.output(38,GPIO.LOW)
            GPIO.output(40,GPIO.HIGH)
            p.ChangeDutyCycle(12.5)

            #send the confidence level to ubidots
            payload = build_payload("confidence",confid_send,"outros",1)
            post_request(payload)
            #time.sleep(2)

        #print the confidence and the name on the image
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confid_show), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    #show the captured image and treated image
    cv2.imshow('camera',img)

    #wait for break
    k = cv2.waitKey(10) & 0xff #press 'ESC' for exiting video
    if k == 27:
        break

    #clean images
    rawCapture.truncate(0)
    
#do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
