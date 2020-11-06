#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Importing libraries
import tensorflow as tf
import os
import pickle
import numpy as np

from flask import Flask, request, redirect, url_for, render_template
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image

OUTPUT_DIR = 'uploads'
SIZE = 64 # Image width

# Setting up environment
if not os.path.isdir(OUTPUT_DIR):
    print('Creating static folder..')
    os.mkdir(OUTPUT_DIR)

app = Flask(__name__)

# To setup Bootstrap templates
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = OUTPUT_DIR

# Model saved with Keras model.save()
with open("model/classifier.pickle", "rb") as handle:
    classifier = pickle.load(handle)
classifier.trainable = False
classifier.compile = True

params = classifier.get_params()
cs = classifier.classes_
print(cs)

my_classes = ['0','1','2','3','4','5','6','7','8','9']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # Check if no file was submitted to the HTML form
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output = make_prediction(filename)
            path_to_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            result = {
                'output': output,
                'path_to_image': path_to_image,
                'size': SIZE
            }
            return render_template('show.html', result=result)
    return render_template('index.html')

def allowed_file(filename):
    '''
    Checks if a given file `filename` is of type image with 'png', 'jpg', or 'jpeg' extensions
    '''
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def make_prediction(filename):
    '''
    Predicts a given `filename` file
    '''
    print('Filename is ', filename)
    fullpath = os.path.join(OUTPUT_DIR, filename)

    # Reshape Function
    test_data = rbgToPixelIntensities(fullpath)
    print(test_data.shape)
    print(type(test_data))
    print(test_data)

    # Generate predictions
    predictions = classifier.predict(test_data)
    print(type(predictions))
    print(predictions)

    # Generate arg maxes for predictions
    print(my_classes[np.argmax(predictions[0], axis=-1)])

    return predictions

def rbgToPixelIntensities(fullpath: np.array) -> np.array:
    """ Convert images in RGB format (w x h x 3) to pixel intensities (w x h)
    Arguments:
    image (numpy.array): an input image in RGB format
    Returns:
    numpy.array: the input image expressed as grayscale pixel intensities
    """
    my_image = image.load_img(fullpath, target_size=(1, SIZE, SIZE))
    my_image = image.img_to_array(my_image)
    print(type(my_image))
    print(my_image.shape)

    scaled = (255 - my_image) / 255
    return np.sqrt(scaled[:, :, 0] ** 2 + scaled[:, :, 1] ** 2 + scaled[:, :, 2] ** 2)

if __name__ == "__main__":
    app.run(debug=True)
