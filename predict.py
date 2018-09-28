import cv2
import json
import keras
import os
import utils
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.optimizers import Adadelta, Adam
from keras.models import load_model
from matplotlib import pyplot as plt

IMG_PATH = os.getenv('IMG_PATH')

def get_model(model_name):
    with open(model_name+ '.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.load_weights(model_name+'.h5')
    return model

# @param image of shape (28, 28, 1)
def predict_input(model, image, classes):
    image = image[np.newaxis,:] # Add dimension to image to fit the input of the model
    prediction = np.argmax(model.predict(image))
    label = classes[prediction]
    return label

# Load the trained model
#model = get_model('test_model')
model = load_model('test_model.model')

# Define classes - {0: 'zero', 1: 'one', 2: 'two', ...}
classes = dict(enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))

# Predict output based on input image
if(IMG_PATH):
    image = cv2.imread(IMG_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert image: black becomes white and white becomes black
    image = utils.invert_image(image)
    image = np.resize(image, (28, 28, 1))
    predicted_label = predict_input(model, image, classes)
    print('Predicted output = ' +predicted_label)
else:
    print('Please give path of image: IMG_PATH=<path> python3 predict.py')

# Testing purposes
#mnist = tf.keras.datasets.mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#predicted_label = predict_input(model, X_test[1], classes)
