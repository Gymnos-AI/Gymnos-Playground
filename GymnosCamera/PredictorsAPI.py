#I had to move this file into this directory because my computer was not working
from flask import Flask, request, Response
import os
import sys
import cv2
import numpy as np
import jsonpickle
import time
import scipy.misc
import Predictors as pred
import tensorflow as tf
from keras import backend as K

app = Flask(__name__)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
predictor = pred.Predictors('YOLOV3')
graph = tf.get_default_graph()

@app.route('/hello', methods=['GET'])
def helloIndex():
    return 'Hello World from Python Flask!'


@app.route('/whothis', methods=['GET'])
def whoIndex():
    return 'YOOOO'

# route http posts to this method
@app.route('/yolov3', methods=['POST'])
def test():
    global graph
    with graph.as_default():
        r = request
        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)
        # decode image  
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        scipy.misc.imsave('image.jpg', img)
        #img.save("image.jpg")
        #cv2.imshow("image", img)
        #print(img.shape)
        list_of_coords = predictor.yolo_v3_detector(img)
        # build a response dict to send back to client
        response = {'coords': list_of_coords.tolist()}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)

        return Response(response=response_pickled, status=200, mimetype="application/json")

app.debug = True
app.run(host='0.0.0.0', port=5000)
