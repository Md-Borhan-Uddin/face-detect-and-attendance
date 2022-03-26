import cv2
import face_recognition
import numpy as np

img = face_recognition.load_image_file("./image/borhan.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faceLoc = face_recognition.face_locations(img)[0]
faceEncode = face_recognition.face_encodings(img)[0]
# print(faceLoc)
# print(faceEncode)
cap = cv2.VideoCapture(0)
while (True):
    _, frame = cap.read()
    imgs = cv2.resize(frame,(0,0),None,0.25, 0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    vLoc = face_recognition.face_locations(imgs)
    vEncode = face_recognition.face_encodings(imgs)

    for code, loc in zip(vEncode,vLoc):
        match = face_recognition.compare_faces([faceEncode], code)
        dis = face_recognition.face_distance([faceEncode], code)
        index = np.argmin(dis)

        if match[index]:
            y1,x2,y2,x1 = loc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            print(y1,x2, x1, y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(frame,(x1,y1),(x2,y2-200),(255,0,0),cv2.FILLED)
            
            
        

    cv2.imshow("image",frame)
    # print(imgs)
   
    
    # result = face_recognition.compare_faces([faceEncode], vEncode)
    # cv2.rectangle(imgs,(vLoc[0],vLoc[3]),(vLoc[1], vLoc[2]),(255,0,0),3)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()