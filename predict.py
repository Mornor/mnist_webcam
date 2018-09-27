import json
import keras
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.optimizers import Adadelta, Adam
from keras.models import load_model
import cv2
import numpy as np
#from matplotlib import pyplot as plt

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

# Predict output based on image
#image = cv2.imread("data/8.png", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("data/3.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = np.resize(image, (28, 28, 1))
print(image.shape)
predicted_label = predict_input(model, image, classes)
print(predicted_label)

#mnist = tf.keras.datasets.mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#
#print(X_test[23].shape)
#predicted_label = predict_input(model, X_test[23], classes)
#print(predicted_label)
