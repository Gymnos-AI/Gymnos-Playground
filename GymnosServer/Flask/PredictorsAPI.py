from flask import Flask, request, Response
import cv2
import numpy as np
import jsonpickle
import time
import GymnosCamera.Predictors as pred
from keras import backend as K


app = Flask(__name__)

predictor = pred.Predictors('YOLOV3')

@app.route('/hello', methods=['GET'])
def helloIndex():
    return 'Hello World from Python Flask!'


@app.route('/whothis', methods=['GET'])
def whoIndex():
    return 'YOOOO'

# route http posts to this method
@app.route('/yolov3', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print(img.shape)
    list_of_coords = predictor.yolo_v3_detector(img)

    # build a response dict to send back to client
    response = {'message': 'yo'}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

app.debug = True
app.run(host='0.0.0.0', port=5000)
