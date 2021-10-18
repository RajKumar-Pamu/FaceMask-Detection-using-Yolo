#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import numpy as np
# In[12]:


yolo = cv2.dnn.readNet("yolov4-custom.cfg","yolov4-custom_best.weights")

cap =cv2.VideoCapture(0)


# In[13]:


classes =[]
with open("./obj.names","r") as f:
    classes =f.read().splitlines()


# In[14]:


while True:
    _,img = cap.read()
    height,width,_=img.shape
    blob =cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
    yolo.setInput(blob)
    op_layer = yolo.getUnconnectedOutLayersNames()
    layer_op = yolo.forward(op_layer)
   
    boxes =[]
    confidences =[]
    class_ids =[]

    for op in layer_op:
        for det in op:
            score = det[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                c_x= int(det[0]*width)
                c_y= int(det[0]*height)
                w= int(det[0]*width)
                h= int(det[0]*height)
                x=int(c_x-w/2)
                y=int(c_y-h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    font  = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    #img = cv2.resize(img,(320,320),interpolation=cv2.INTER_AREA)

    if len(boxes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            con = str(confidences[i])
            col = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),col,2)
            cv2.putText(img,label+" "+con,(x,y+20),font,2,(255,255,255),1)
    
    cv2.imshow("live",img)
    key=cv2.waitKey(1)
    if key==27:
        cap.release()
        cv2.destroyAllWindows()
        break


# In[ ]:




