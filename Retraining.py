import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
path = 'dataset'
#set the face classifier model
recognizer = cv2.face.LBPHFaceRecognizer_create(neighbors = 10)
#neighbors = 20, grid_x = 20, grid_y = 20
#set the face detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
# function to get the images and label data
def getImagesAndLabels(path):
    #set the image paths, for alll images to be classified
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
#set the training data for training
faces,ids = getImagesAndLabels(path)
#retrain the model
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer_default.yml') # save the trainned model
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))