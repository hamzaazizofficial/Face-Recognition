#Face Detection p1
#importing library
import numpy as np
import cv2

#getting classifier file
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#extracting the features of face
def face_extractor(img):
    #converting RGB img into GrayScale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #detectMultiScale takes src, scaling fac, min neighbors (from 3-6)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #making some conditions
    if faces is():
        return None

    #if faces() have some val
    #passing X & Y cord and Height & Width
    for(x,y,w,h) in faces:
        #cropping faces
        cropped_faces = img[y:y+h, x:x+w]
    
    return cropped_faces



#configuring the camera
cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
     #condition 
    if face_extractor(frame) is not None:
         count+=1
         #resizing face.. cv2.resize takes src, resizing val in tuple
         face = cv2.resize(face_extractor(frame),(200,200))
         #converting the resized img into Grayscale
         face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
         #save the val(s) of these faces
         file_name_path = 'C:/Users/HP/AppData/Local/Programs/Python/Python36/OpenCV/faces/user'+str(count)+'.jpg'
         #saving the imgs
         cv2.imwrite(file_name_path,face)
         #counting the imgs... cv2.putText() takes src,count,origin point,format,scaling fact,color,thickness
         cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
         cv2.imshow('face Cropper',face)
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break

cam.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete!")
