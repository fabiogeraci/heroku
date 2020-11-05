#Python Standard lib
import numpy as np
import os
import pickle
from waitress import serve

#Flask Import for web app
from flask import Flask, render_template, request, redirect, flash, url_for
import urllib.request
from werkzeug.utils import secure_filename

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Model saved with Keras model.save()
with open("model/classifier.pickle", "rb") as handle:
    classifier = pickle.load(handle)
#MODEL_PATH = 'malaria_detector.pkl'

# Load your own trained model
#model = load_model(classifier)
#print('Model loaded. Start serving...')

MYDIR = os.path.dirname(__file__)
UPLOAD_FOLDER = 'images/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_prediction(filename):

    #image = load_img('images/' + filename, target_size=(224, 224))
    image = load_img(os.path.join(app.config['UPLOAD_FOLDER'], "image432.jpeg"), target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    classifier.precompute = False
    yhat = classifier.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    print('%s (%.2f%%)' % (label[1], label[2] * 100))
    return label[1], label[2] * 100

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/", methods=['GET','POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            get_prediction(filename)
            label, acc = get_prediction(filename)
            flash(label)
            flash(acc)
            flash(filename)
            return redirect('/')


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=port)
#    app.run()