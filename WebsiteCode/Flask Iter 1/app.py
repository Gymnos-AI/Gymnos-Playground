from flask import Flask, render_template, request
from numpy as np
import keras.models
import re
import sys
import os
sys.path.append(os.path.abspath('./model'))
from load import *

#init flash app
app = Flask(__name__)

global model, graph
model, graph = init()

#decoding an image from base64 into raw representation
def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		output.write(imgstr.decode('base64'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', method=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x = imread('out.png', mode ='L')
    x = np.invert(x)
    x = imresize(x, 256, 256)
    x = x.reshape(1, 256, 256, 1)
    with graph.as_default():
        out = model.predict(x)
        reponse = np.array_str(np.argmax(out, axis=1))
        return response

if __name__ = '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
