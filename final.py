#For LBPH input and output can only be integers 
#LBPH needs BnW photos for training
#We have generated a random integer for every name you enter, stored this as key(ID):value(Name) pair in data.json 
#Unknown is not a label, any face that is not recognized is an Unknown face

import cv2
import numpy as np
import os
import serial
import json
import random
import sys
from PIL import Image
from statistics import mode
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
data = {}  
data['people'] = [] 
with open('datax.json', 'w') as outfile:  #dictionary for mapping ids with names
    json.dump(data, outfile)
recognizer = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists("saved_model/")
recognizer.read('saved_model/s_model.yml')
DetectPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(DetectPath);
font = cv2.FONT_HERSHEY_SIMPLEX

emotion_model_path = '../trained_models/emotion_models/Emo.hdf5'
emotion_labels = get_labels('fer2013')

frame_window = 1
emotion_offsets = (20, 40)

face_detection = load_detection_model(DetectPath)
emotion_classifier = load_model(emotion_model_path, compile=False)

emotion_target_size = emotion_classifier.input_shape[1:3]

emotion_window = []

cv2.namedWindow('FR+Emo')
video_capture = cv2.VideoCapture(0)
while 1:
    a=0     #a=(100-confidence)
    an =0   #number of angry faces in a single frame 
    ha = 0  #number of happy faces in a single frame
    s =0    #number of sad faces in a single frame
    su =0   #number of surprised faces in a single frame
    bgr_image = video_capture.read()[1]      #check#
    #name ="Unknown"
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)
    k=cv2.waitKey(30) & 0xff
    #s = serial.Serial('/dev/ttyUSB0') 
    #ser = 0
    #ser = serial.Serial('/dev/ttyUSB0', timeout=None, baudrate=115200, xonxoff=False, rtscts=False, dsrdtr=False)
    
    for face_coordinates in faces:
        #name ="Unknown"
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
	'''
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            an+=1
            color = emotion_probability * np.asarray((0, 0, 0))
        elif emotion_text == 'sad':
            s+=1
            color = emotion_probability * np.asarray((0, 0, 0))
        elif emotion_text == 'happy':
            ha+=1
            color = emotion_probability * np.asarray((0, 0, 0))
        elif emotion_text == 'surprise':
            su+=1
            color = emotion_probability * np.asarray((0, 0, 0))
        else:
            emotion_text== 'neutral'
            color = emotion_probability * np.asarray((0, 0, 0))

        color = color.astype(int)
        color = color.tolist()

        #draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_text,
                  (255,0,0), 0, +45, 1, 1)

    for(x,y,w,h) in faces:
        #name = "Unknown"
        cv2.rectangle(rgb_image, (x,y), (x+w,y+h), (0,0,0), 4)
        Id, confidence = recognizer.predict(gray_image[y:y+h,x:x+w]) #ids corresponding to confidence
        with open('data.json') as f:
             data = json.load(f)['people']   #reading from the data.json file for names linked with the ids
        p= str(Id)

        #name = "Unknown"
        for person in data:
            if p in person:
                name = person[p]
                         
        #with open('datax.json') as outfile:
              #data=json.load(outfile)
              #data['people'].append({  
                #name : emotion_text
              #})

        #with open('datax.json', 'w') as outfile:  
            #json.dump(data, outfile)
        a = round(100-confidence,2)     #rounding off the a value to 2 digits after the decimal
       
        if a >=17:        #fine tuning needed for filtering false positive results, works fine for 15-20
          cv2.rectangle(rgb_image, (x-2,y-70), (x+w+2, y), (0,0,0), -1)
          cv2.putText(rgb_image, name, (x,y-40), font, 1, (255,255,255), 3)
        else:
          #a = 0
          cv2.rectangle(rgb_image, (x-2,y-70), (x+w+2, y), (0,0,0), -1)
          cv2.putText(rgb_image, "Unknown", (x,y-40), font, 1, (255,255,255), 3)
          #name = "unknown"
          #if cv2.waitKey(10000) or 0xFF == ord('q'):
              #break
          #s=serial.Serial("/dev/ttyUSB0","115200")
          #s.write(name.encode('ascii'))
          #break
          #print(name)
          #cv2.waitKey(10000)
    cv2.putText(rgb_image, str(an)+" "+str(s)+" "+str(su)+" "+str(ha)+" ", (0,40), font, 1, (255,255,255), 3)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)  
    #ser = serial.Serial('/dev/ttyUSB0', 105200, timeout=1)
    #while True:
       #name = ser.readline()
       #break
    #break
    #print(name)      
    
    if a>95:
     #s=serial.Serial("/dev/ttyUSB0","115200")
     #s.write(name.encode('ascii'))
     print(name)
     #hello(name)
     video_capture.release()
     cv2.destroyAllWindows()
     sys.exit()
     
    #ser = serial.Serial('/dev/ttyUSB0',"115200", timeout=0)
    #x= ser.readline()
    #if len(x)>0:
     
     #ser=serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
     #name = x.decode("utf-8") 
     #video_capture.release()
     #cv2.destroyAllWindows()
     #print(name)
     #break
    if cv2.waitKey(1) & 0xFF == ord('q'):
     sys.exit()
    
      
    #print(emotion_text)
    #break
 
    #break
 #if a>30:
    #break
 #break





#1Taking photos for Training

for m in range(1):
  id = random.randint(1,100000)
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

video_capture = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#name = input("Enter Name: ")
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

    
    _, image_frame = video_capture.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:

        
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)###
        count += 1
        cv2.imwrite("training_data/X." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame', image_frame)

    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif count>49:
        break


video_capture.release()
cv2.destroyAllWindows()




#2 Training




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
#def hello():
 #video_capture.release()
 #cv2.DestroyAllWindows()


#print(s=serial.Serial("/dev/ttyUSB0","105200")
        #break
#video_capture.release()
#cv2.DestroyAllWindows()
      

#s=serial.Serial("/dev/ttyUSB0","105200")
#s.write(name.encode('ascii'))
#print(s=serial.Serial("/dev/ttyUSB0","105200")
#print(name)	
#print (emotion_text)
