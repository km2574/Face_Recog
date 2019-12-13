import cv2
import os
import json
import numpy as np
from PIL import Image
import random
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for m in range(1):
  id = random.randint(1,100000)
name = input("Enter Name: ")
with open('data.json') as f:
 	data=json.load(f)
 	data['people'].append({
     		str(id) : name,
	})
with open('data.json', 'w') as f:  
  	json.dump(data, f)

count = 0
assure_path_exists("training_data/")
while(True):

    
    _, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:

        
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("training_data/X." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame', image_frame)

    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif count>29:
        break


vid_cam.release()
cv2.destroyAllWindows()



def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    
    return faceSamples,ids


faces,ids = getImagesAndLabels('training_data')
recognizer.train(faces, np.array(ids))
assure_path_exists('saved_model/')
recognizer.write('saved_model/s_model.yml')

