# Set of tools to perform operations on dataset
# @author Celien Nanson <cesliens@gmail.com>

# Import
import json
import cv2
import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt

# Load and return the MNIST dataset from tf.keras
# 60000 images and labels
# Each image is 28x28 pixels
def load_dataset():
    mnist = tf.keras.datasets.mnist
    return mnist.load_data()

def invert_image(image):
    return cv2.bitwise_not(image)

# Save the model under ./model.json, as well as the weights under ./model.h5
def save_model(model, name):
    with open('./' +name+ '.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    model.save_weights(name + '.h5')
