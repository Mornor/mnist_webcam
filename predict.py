import json
import keras
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.optimizers import Adadelta, Adam
from PIL import Image

def get_model():
    with open('model.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))

    optimizer = Adadelta()
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.load_weights('model.h5')
    return model


def predict_input(model, image):
    preditcted_number = int(model.predict(image, batch_size=1))

# Load the trained model
model = get_model()

# Predict output based on image
image = Image.open("data/3.png")
predicted_number = predict_input(model, image)