# Set of tools to perform operations on dataset
# @author Celien Nanson <cesliens@gmail.com>

# Import
import json
import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt

# Load and return the MNIST dataset from tf.keras
# 60000 images and labels
# Each image is 28x28 pixels
def load_dataset():
    mnist = tf.keras.datasets.mnist
    return mnist.load_data()

# Display the images[index] to the console
def display_image(images, index):
    print(mndata.display(images[index]))

# Display the labels[index] to the console
def display_label(labels, index):
    print(labels[index])

# Save the model under ./model.json, as well as the weights under ./model.h5
def save_model(model, name):
    with open('./' +name+ '.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    model.save_weights(name + '.h5')

#(X_train, y_train), (X_test, y_test) = load_dataset()
#plt.imshow(X_test[0])
#plt.show()
#images, labels = load_test_set()
#display_image(images, 235)
