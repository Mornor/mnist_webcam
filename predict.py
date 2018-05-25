import json
import keras
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.optimizers import Adadelta, Adam
import cv2
import numpy as np

def get_model(model_name):
    with open(model_name+ '.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))

    optimizer = Adadelta()
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.load_weights('model.h5')
    return model

def img_to_mnist(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=321, C=28)
    return gray_img

def predict_input(model, image, classes):
    class_prediction = model.predict_classes(image)[0]
    prediction = np.around(np.max(model.predict(image)), 2)
    label = classes[class_prediction]
    print(label)


# Load the trained model
model = get_model('test_model')

# Define classes - {0: 'zero', 1: 'one', 2: 'two', ...}
classes = dict(enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))

# Predict output based on image
image = cv2.imread("data/3.png")
image = img_to_mnist(image) # (351, 353)
image = image.reshape(1, image.shape[0], image.shape[1], 1) # transform into (1, 351, 2353, 1)
#print(image.shape)
predicted_number = predict_input(model, image, classes)
