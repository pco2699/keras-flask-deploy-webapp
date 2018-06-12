from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# OpenCV
import cv2

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/gs_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')

def faceDetectionFromPath(path, size):
    cvImg = cv2.imread(path)
    cascade_path = "./lib/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(cvImg, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    faceData = []
    for rect in facerect:
        faceImg = cvImg[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        resized = cv2.resize(faceImg,None, fx=float(size/faceImg.shape[0]),fy=float( size/faceImg.shape[1]))
        CV_im_RGB = resized[:, :, ::-1].copy()
        pilImg=Image.fromarray(CV_im_RGB)
        faceData.append(pilImg)
 
    return faceData
 
img_rows, img_cols = 100, 100

def model_predict(img_path, model):
    faceImgs = faceDetectionFromPath(img_path, img_rows)
    x = []
    for faceImg in faceImgs:
        x.append(image.img_to_array(faceImg))

    # Preprocessing the image
    x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        res = ['ジーズ生', 'ジーズ生じゃない']
        # Make prediction
        preds = model_predict(file_path, model)
        for pred in preds:
            predR = np.round(pred)
            for pre_i in np.arange(len(predR)):
                if predR[pre_i] == 1:
                    result = res[pre_i]

        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
