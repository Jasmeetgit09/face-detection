import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase,webrtc_streamer
from keras.preprocessing.image import load_img
import numpy as np
from keras.models import load_model
##from tensorflow.keras.utils import img_to_array


st.title("Face Emotion Detection")
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

classifier =load_model('Final_model.h5')  #Load model


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray)
        emotion_list = [ 'angry' , 'disgust' , 'fear' , 'happy' , 'neutral' , 'sad' , 'surprise']
        
        

        i =self.i+1
        for (x, y, w, h) in faces:
            image = img[y:y+h, x:x+w]
                      
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), (95, 207, 30), -1)
            # cv2.putText(img, '{}'.format(emotion_list[predicted_value]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0) ## reshaping the cropped face image for prediction

                prediction = classifier.predict(roi)[0]   #Prediction
                label=emotion_list[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)   # Text Adding
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        return frame

webrtc_streamer(key="example", video_processor_factory=VideoTransformer)
