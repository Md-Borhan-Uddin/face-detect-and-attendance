import requests
import json
import cv2

res = requests.get("http://127.0.0.1:8000/")

data = res.json()

for d in data:

    r = cv2.imread("http://127.0.0.1:8000"+d['img'])
    print(r)