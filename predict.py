import json
import keras
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.optimizers import Adadelta, Adam
#import cv2
import numpy as np
#from matplotlib import pyplot as plt

def get_model(model_name):
    with open(model_name+ '.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.load_weights('model.h5')
    return model

def img_to_mnist(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

def predict_input(model, image, classes):
    class_prediction = model.predict_classes(image)[0]
    prediction = np.around(np.max(model.predict(image)), 2)
    label = classes[class_prediction]
    print(label)


# Load the trained model
model = get_model('model')

# Define classes - {0: 'zero', 1: 'one', 2: 'two', ...}
classes = dict(enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

predictions = model.predict([X_test])
#print(predictions)
print(np.argmax(predictions[0]))

# Predict output based on image
#image = cv2.imread("data/3.png", cv2.IMREAD_GRAYSCALE)
#image = np.resize(image, (image.shape[0], image.shape[1], 1))
#print(image.shape)
#plt.imshow(image, cmap='gray')
#plt.show()

#print(image.shape)
#image = cv2.imread("data/3.png", cv2.IMREAD_GRAYSCALE)
#image = img_to_mnist(image) # (351, 353)
#print(image.shape)
#image = image[np.newaxis,:] # transform into (1, :, :, :)
#print(image.shape)
#image = image.reshape(28, 28, 1) # transform into (1, 351, 353, 1)
#print(image.shape)
#predicted_number = predict_input(model, image, classes)
