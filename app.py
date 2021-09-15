import flask
from flask.helpers import send_file
from flask.wrappers import Response
import numpy as np
from skimage import io
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot  as plt
from flask import Flask, request, render_template, session, redirect
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import sys
from scipy.spatial import distance

app = flask.Flask(__name__, template_folder='templates')

def testing(name):
    model = load_model("classify_model.h5")
    face_model = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
            
    mask_label = {0:'Pakai Masker',1:'Tanpa Masker'}
    dist_label = {0:(0,255,0),1:(255,0,0)}
    MIN_DISTANCE = 0

    img= io.imread(name)

    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=8)


    if len(faces)>=1:
        label = [0 for i in range(len(faces))]
        for i in range(len(faces)-1):
            for j in range(i+1, len(faces)):
                dist = distance.euclidean(faces[i][:2],faces[j][:2])
                if dist<MIN_DISTANCE:
                    label[i] = 1
                    label[j] = 1
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            crop = new_img[y:y+h,x:x+w]
            crop = cv2.resize(crop,(150,150))
            crop = np.reshape(crop,[1,150,150,3])/255.0
            mask_result = model.predict(crop)
            cv2.putText(img,mask_label[round(mask_result[0][0])],(x, y+90), cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
            cv2.rectangle(img,(x,y),(x+w,y+h),dist_label[label[i]],1)
        
        plt.imshow(img) #Needs to be in row,col order
        plt.savefig("./static/result.png")
                
    else:
        print("No Face!")

    return redirect("/static/result.png", code=302)



@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    return testing(uploaded_file.filename)

@app.route('/about')
def about():    
    return flask.render_template('about.html')

@app.route('/')
def main():
    return(flask.render_template('main.html'))

if __name__ == '__main__':
    app.run()
