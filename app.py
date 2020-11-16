#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Importing libraries
import tensorflow as tf
import os
import time
import numpy as np
import pickle

# load and show an image with Pillow
from PIL import Image

# import flask, flask_bootstrap, werkzeug
from flask import Flask, request, redirect, url_for, render_template
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

# setting up folder structure for deployment
OUTPUT_DIR = 'uploads'
DOWNLOAD_DIR = "_static/images"
if not os.path.isdir(OUTPUT_DIR):
    print('Creating static folder..')
    os.mkdir(OUTPUT_DIR)

# Image width & length
img_size = 8

app = Flask(__name__)

# To setup Bootstrap templates
Bootstrap(app)
app.config['secret_key'] = "myownsecretkey"
app.config['UPLOAD_FOLDER'] = OUTPUT_DIR
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_DIR

# Load trained model
with open("model/classifier.pickle", "rb") as handle:
    classifier = pickle.load(handle)


@app.route('/', methods=['GET', 'POST'])
def load_image():
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
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['DOWNLOAD_FOLDER'], filename))
            print(os.path.join(app.config['DOWNLOAD_FOLDER'], filename))

            image = Image.open(os.path.join(app.config['DOWNLOAD_FOLDER'], filename))
            output = make_prediction(image)
            path_to_image = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
            result = {
                'output': output,
                'path_to_image': path_to_image,
                'size': 200
                }
            return render_template('show.html', result=result)
    return render_template('index.html')


def make_prediction(image):

    scaled_img = rbg_to_pixel_intensities(image)
    transformed_img = img_reshape(scaled_img)

    # Generate predictions
    predictions = classifier.predict(transformed_img)
    print('This is the given prediction' +
          ' = ' + str(predictions))
    return predictions


def rbg_to_pixel_intensities(image):
    print('Image Format' + ' = ' + str(image.format))
    print('Image Mode' + ' = ' + str(image.mode))
    print('Image Size' + ' = ' + str(image.size))

    # Image resize to train model
    resize_img = image.resize((img_size, img_size))
    print('New Image Size' + ' = ' + str(resize_img.size))

    img_to_array = np.array(resize_img)
    scaled = (255 - img_to_array) / 255
    print('scaled shape' + ' = ' + str(scaled.shape))
    print('scaled max' + ' = ' + str(scaled.max()))
    print('scaled min' + ' = ' + str(scaled.min()))

    return np.sqrt(scaled[:, :, 0] ** 2 + scaled[:, :, 1] ** 2 + scaled[:, :, 2] ** 2)


def img_reshape(scaled):
    print('Entering Scaled Image' + ' = ' + str(scaled.shape))

    # Reshape array to fit training model
    transformed_img = scaled.reshape(1, 64)
    transformed_img = np.interp(transformed_img, (transformed_img.min(), transformed_img.max()), (0, 16))
    print('transformed_img shape' + ' = ' + str(transformed_img.shape))
    print('transformed_img shape' + ' = ' + str(transformed_img.max()))
    print('transformed_img shape' + ' = ' + str(transformed_img.min()))
    return transformed_img


if __name__ == '__main__':
    app.run(debug=True)
