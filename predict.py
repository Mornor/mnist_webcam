import json
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.optimizers import Adadelta, Adam

def get_model():
    with open('model.json', 'r') as jfile:
        model = model_from_json(json.load(jfile))

    optimizer = Adadelta()
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.load_weights('model.h5')
    return model

# Load the trained model
model = get_model()