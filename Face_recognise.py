#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import os


# In[3]:


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\\Users\\Daman\\Machine Learning\\haarcascade_frontalface_alt.xml')
face_data = []
labels = []
name = {}
face_section = np.zeros((100,100),dtype = "uint8")
class_id = 0
path = "C:\\Users\\Daman\\Desktop\\Faces"

for file in os.listdir(path):
    if file.endswith(".npy"):
        content = np.load(path+'\\'+file)
        name[class_id] = file[:-4]
        face_data.append(content)
        target = class_id * np.ones((content.shape[0],))
        class_id = class_id + 1
        labels.append(target)
        
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape(-1,1)

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key = lambda x: x[2]*x[3])
    
    for face in faces[-1:]:
        x,y,w,h = face
        offset = 10
        face_section = gray[x-offset : x+w+offset , y-offset : y+h+offset ]
        face_section = cv2.resize(face_section,(100,100)) 
        pred = KNN(face_dataset,face_labels,face_section.flatten())
        text = name[int(pred)]
        cv2.putText(frame,text,(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
    cv2.imshow('frame',frame)
        
    keypressed = cv2.waitKey(1) & 0xff
    if(keypressed == ord('q')):
        break
        
cap.release()
cv2.destroyAllWindows()
        


# In[1]:


def KNN(X,y,point,k=5):
    val = []
    m = X.shape[0]
    for i in range(m):
        xi = X[i]
        #xi = xi.reshape((64,))
        dist = distance(x , xi)
        val.append((dist,y[i]))
        
    vall = sorted(val,key= lambda x:x[0])[:k]
    val = np.array(vall)
    val_point = np.unique(val[:,1], return_counts = True)
    index = val_point[1].argmax()
    result = val_point[0][index]
    return result

def distance (x,xi):
    dist = np.sqrt(np.sum((x - xi)**2))
    return dist


# In[ ]:




