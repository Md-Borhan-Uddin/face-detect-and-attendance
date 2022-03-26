import json
import cv2
import datetime
import requests
import os
import numpy as np
import face_recognition

path = 'img'
images = []
image_url = []
person_name = []


URL = "http://127.0.0.1:8000/"

res = requests.get(URL)

data = res.json()

for d in data:
    image_url.append("http://127.0.0.1:8000"+d['img'])

print(image_url)
j=1
for i in image_url:
    res = requests.get(i)
    with open(f"./img/sinple_{j}.jpg", 'wb') as f:
        f.write(res.content)
    j+=1

image_list = os.listdir(path)
for img in image_list:
    currentimg = face_recognition.load_image_file(f"{path}/{img}")

    images.append(currentimg)
    person_name.append(os.path.splitext(img)[0])


def get_encode(img_list):
    encodelist = []
    for img in img_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img)
        enc = face_recognition.face_encodings(img)[0]
        encodelist.append(enc)
    return encodelist


knownencodelist = get_encode(images)
print("end Encoding")


cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    imgs = cv2.resize(frame,(0,0), None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    currentigmloc = face_recognition.face_locations(imgs)
    currentimgecode = face_recognition.face_encodings(imgs, currentigmloc)

    for imgecode, imgloc in zip(currentimgecode, currentigmloc):
        match = face_recognition.compare_faces(knownencodelist, imgecode)

        distence = face_recognition.face_distance(knownencodelist, imgecode)

        matchindex  = np.argmin(distence)

        if match[matchindex]:
            name = person_name[matchindex]
            y1,x2,y2,x1 = imgloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            print(y1,x2, x1, y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(frame,(x1,y1),(x2,y2-200),(255,0,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y1+6), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            time = datetime.datetime.now()
            id = matchindex+1
            data = {
                "name" : name,
                "e_id":str(id),
                "date":str(time)
            }
            header = {"content_type":"application/json"}
            data = json.dumps(data)
            r = requests.post(URL, headers = header, data=data)
            # print(r.json())
        else:
            print("unknown")

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

