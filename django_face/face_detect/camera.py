import cv2
import datetime
import numpy as np
import os
import face_recognition
from django.conf import settings



class Camera(object):
    def __init__(self,path):
        self.path = path
        self.images = []
        self.person_name = []
        self.knownencodelist = []
    
    def __del__(self):
        cv2.destroyAllWindows()
    
    def get_img_list(self):
        img_list = []
        image_list = os.listdir(self.path)
        for img in self.image_list:
            currentimg = face_recognition.load_image_file(f"{self.path}/{img}")

            self.images.append(currentimg)
            self.person_name.append(os.path.splitext(img)[0])
        return img_list

    def get_encode(self):
        encodelist = []
        for img in self.get_img_list():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(img)
            enc = face_recognition.face_encodings(img)[0]
            encodelist.append(enc)
        return encodelist


    
    print("end Encoding")

    def get_frame(self):

        cap = cv2.VideoCapture(0)

        while True:
            success, frame = cap.read()
            imgs = cv2.resize(frame,(0,0), None,0.25,0.25)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            currentigmloc = face_recognition.face_locations(imgs)
            currentimgecode = face_recognition.face_encodings(imgs, currentigmloc)

            for imgecode, imgloc in zip(currentimgecode, currentigmloc):
                match = face_recognition.compare_faces(self.get_encode, imgecode)

                distence = face_recognition.face_distance(self.get_encode, imgecode)

                matchindex  = np.argmin(distence)

                if match[matchindex]:
                    name = self.person_name[matchindex]
                    y1,x2,y2,x1 = imgloc
                    y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                    print(y1,x2, x1, y2)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.rectangle(frame,(x1,y1),(x2,y2-200),(255,0,0),cv2.FILLED)
                    cv2.putText(frame,name,(x1+6,y1+6), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                else:
                    print("unknown")
            fes, jpg = cv2.imencode('.jpg', frame)
            return jpg.tobytes()
            

        
    