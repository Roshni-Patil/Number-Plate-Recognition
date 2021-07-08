from flask import Flask , render_template , request
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import easyocr

app = Flask("Car Details")



@app.route("/home")
def home():
        return render_template( "index.html" )


@app.route("/car-details" , methods = ["GET" ] )
def prediction():
        plateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
        x1 = request.args.get( "upload" )
        #detect the plate and return car + plate image
        platePart = 0
        plate_number=0
        def detect_plate_no(img):
            global platePart
            global plate_number
            plateImg = img.copy()
            roi = img.copy()
            plateRect = plateCascade.detectMultiScale(plateImg,scaleFactor = 1.5, minNeighbors = 7)
            for (x,y,w,h) in plateRect:
                roi_ = roi[y:y+h, x:x+w, :]
                platePart = roi[y:y+h, x:x+w, :]
                cv2.rectangle(plateImg,(x+2,y),(x+w-3, y+h-5),(0,255,0),3)
            return plateImg, platePart
        inputImg = cv2.imread(x1)
        inpImg , plate = detect_plate_no(inputImg)
        reader = easyocr.Reader(['en'])
        result = reader.readtext(plate)
        final_result = result[0][1]
        n=""
        f = final_result.split('-')
        f = n.join(f)
        f = f.replace(" ","")
        plate_number = f.upper()
        return plate_number


        

app.run(host="172.17.0.2" , port=8080, debug=True)

