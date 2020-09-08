import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

def predictperson():
    video_capture = cv2.VideoCapture(0)
    while(True):
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        ret,frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
        height, width, channels = frame.shape
        
        faces_inside_box = 0
        
        for (x, y, w, h) in faces:
            faces_inside_box+=1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        if faces_inside_box > 1:
            cv2.putText(frame,"Multiple Faces detected!", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if faces_inside_box == 1:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            image = cv2.resize(frame, (128, 128))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            (real, fake) = model.predict(image)[0]
            
            if fake > real:
                label = "real"
            else:
                label= "fake"
            label = "{}".format(label)
            cv2.putText(frame,label, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Frame",frame)

model = load_model("models/the-unreal.hdf5")
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
predictperson()