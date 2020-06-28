#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np


# In[4]:


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("C:\\Users\\Daman\\Machine Learning\\haarcascade_frontalface_alt.xml")
face_data = []
path = "C:\\Users\\Daman\\Desktop\\Faces"
face_section = np.zeros((100,100),dtype="uint8")
skip = 0
name = input('enter name')
while True:
    ret,frame = cap.read()
    grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(grayscale,1.3,5)
    faces = sorted(faces,key = lambda f:f[2]*f[3])
    
    for face in faces[-1:]:
        x,y,w,h = face
        offset = 10
        face_section = grayscale[ y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        
        
    cv2.imshow("camera",frame)
    key_pressed = cv2.waitKey(1) & 0xff
    if key_pressed == ord('q'):
        break
    if (skip % 10 == 0):
        face_data.append(face_section)
    skip = skip + 1
    
print(face_data)        
face_data = np.asarray(face_data)
print(face_data.shape)
face_data = face_data.reshape((face_data.shape[0],-1))
np.save(path+"\\"+name+".npy",face_data)  
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




