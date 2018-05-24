import json
import keras
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.optimizers import Adadelta, Adam
import cv2

def get_model():
    with open('model.json', 'r') as jfile:
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

def predict_input(model, image):
    preditcted_number = int(model.predict(image, batch_size=1))

# Load the trained model
model = get_model()

# Predict output based on image
image = cv2.imread("data/3.png")
image = img_to_mnist(image)
predicted_number = predict_input(model, image)