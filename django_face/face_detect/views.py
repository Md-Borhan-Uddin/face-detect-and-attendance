import json
import cv2
import datetime
import requests
import os
from PIL import Image
import numpy as np
import face_recognition
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from face_detect.serilizers import profileserializer, Attenserializer
from rest_framework.decorators import api_view
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from face_detect.models import Profile, Attendance
from face_detect.camera import Camera
from django.conf import settings
import io





# Create your views here.
@csrf_exempt
@api_view(['GET','POST'])
def home(request):
    if request.method=="GET":
        p = Profile.objects.all()
        s = profileserializer(p, many=True)

        j = JSONRenderer().render(s.data)
        
        return HttpResponse(j, content_type='application/json')
    
    if request.method=="POST":
        d = request.body
        st = io.BytesIO(d)
        p = JSONParser().parse(st)
        s = Attenserializer(data=p)
        id = int(p.get('e_id'))
        data = Attendance.objects.filter(e_id=id)
        if data.exists():
            pass
        else:
            if s.is_valid():
                s.save()
                return HttpResponse({"msg":"ok"}, content_type="application/json")
        

def gen(request):
    

    path = settings.BASE_DIR/'media/img'
    images = []
    person_name = []



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
        pl_lmg = Image.fromarray(imgs)
        pl_lmg.show("my")
        print(pl_lmg)
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
               
            else:
                print("unknown")

        cv2.imshow("video", frame)
    

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'home1.html')

def camera(request):
    return render(request, 'home.html')



# def facecam_feed(request):
# 	return StreamingHttpResponse(gen(FaceDetect()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')