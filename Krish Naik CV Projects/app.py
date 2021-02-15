from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from numpy import expand_dims
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
#Mongo Engine (Database)
from flask_mongoengine import MongoEngine
from mongoengine import connect, disconnect

# Define a flask app
app = Flask(__name__)

#connect to MongoDB database (make sure its GridFS)
DB_URI= 'mongodb+srv://DataCowboy:Sheridahill20!@testcluster.ksdrr.mongodb.net/ComputerVision?retryWrites=true&w=majority'
app.config['MONGODB_HOST'] = DB_URI
me = MongoEngine(app)

#Define .h5 model path (locally)
MODEL_PATH = r"C:\Users\DataCowboy\Downloads\CycleGAN Models\Models AtoB\g_model_AtoB_002370.h5"

# Load trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

'''
#Create our image class in the database, I also assume this will most likely be used in a function
class Image(me.Document):
    photo= ImageField()

file= Image(photo= 'Upload')
with open('.jpg', 'rb') as fd:
    file.photo.put(fd, content_type= 'image/jpeg')
#save file to database
file.save()

#Retrieve file from database
file= Image.objects(photo= '.jpg').first()
photo= file.photo.read()
content_type= photo.photo.content_type
'''

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/')
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

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

