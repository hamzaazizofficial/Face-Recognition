#Face Detection p2 || Training Model
#importing library
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join 

#getting dataset
dataset = 'C:/Users/HP/AppData/Local/Programs/Python/Python36/OpenCV/faces/'

#getting files
onlyfiles = [f for f in listdir(dataset) if isfile(join(dataset,f))]

training_set, labels = [],[]

for i,files in enumerate(onlyfiles):
    img_path = dataset + onlyfiles[i]
    #reading grayscale imgs
    images = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    #appending training set into array
    training_set.append(np.asarray(images, dtype = np.uint8))
    #appending labels 
    labels.append(i)

labels = np.asarray(labels, dtype = np.int32)



model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(training_set), np.asarray(labels))

print("Model is Trained Succesfully!")

#getting classifier file
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#making face detector
def face_detector(img, size = 0.5):
    #converting img into Gray img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    #if faces are absent
    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        #making rect for detecting
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        #making region of interest.. taking vals rows and cols
        roi = img[y:y+h, x:x+w]
        #resizing new img
        roi = cv2.resize(roi, (200,200))

    return img,roi

#configuring camera
cam = cv2.VideoCapture(0)
while True:

    ret, frame = cam.read()

    image, face = face_detector(frame)

    #making exception handling for checking errors
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #prediction
        result = model.predict(face)

        #checking % of detection
        if result[1] < 500:
            check = int(100*(1-(result[1])/300))
            display_str = str(check)+'%  it is user'
        #displaying text
       # cv2.putText(image,display_str,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        #if face is matched or not
        if check > 75:
            cv2.putText(image, "Face Matched!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Wrong Face!", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        #if face is absent
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break


cam.release()
cv2.destroyAllWindows()

