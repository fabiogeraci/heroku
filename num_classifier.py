import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidml.keras
plaidml.keras.install_backend()
import tensorflow as tf
print('Tensorflow Version'+' = '+ tf.__version__)
import keras

# import other stuff
from keras import backend as K

import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
#%matplotlib inline

print(x_train.shape)
single_image = x_train[0]
#plt.imshow(single_image)
#plt.show()

from tensorflow.keras.utils import to_categorical
y_example = to_categorical(y_train)
print(y_example.shape)

y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape(60000, 8, 8, 1)
x_test = x_test.reshape(10000,8,8,1)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(8, 8, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))

# we can add in additional metrics
# https://keras.io/metrics/
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

from keras.callbacks import EarlyStopping, History
early_stop = EarlyStopping(monitor='val_loss',patience=2)
history = History()
model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop, history])

from keras.models import save_model, load_model

# save model and architecture to single file
filename = 'classifier.pickle'
pickle.dump(model, open(filename, 'wb'))

print(model.metrics_names)

print(model.evaluate(x_test,y_cat_test,verbose=0))

losses = pd.DataFrame(model.history.history)
print(model.history.history.keys())
print(losses.head())

# Create count of the number of epochs
epoch_count = range(1, len(losses) + 1)

plt.plot(epoch_count, losses['acc'], color='blue', label='acc')
plt.plot(epoch_count, losses['val_acc'], color='red', label='val_acc')
plt.ylabel('acc value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
plt.savefig('acc_Vs_valacc.png')
plt.close()

plt.plot(epoch_count, losses['loss'], color='blue', label='loss')
plt.plot(epoch_count, losses['val_loss'], color='red', label='val_loss')
plt.ylabel('loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
plt.savefig('loss_Vs_valloss.png')
plt.close()

print(model.evaluate(x_test, y_cat_test, verbose=0))

#from sklearn.metrics import classification_report, confusion_matrix
#predictions = model.predict_classes(x_test)
#print(classification_report(y_test, predictions))
