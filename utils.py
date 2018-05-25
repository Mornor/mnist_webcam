# Set of tools to perform operations on dataset
# @author Celien Nanson <cesliens@gmail.com>

# Import
import mnist
import json

DATA_DIR_PATH = 'data/'
mndata = mnist.MNIST(DATA_DIR_PATH)

# Load and return the MNIST dataset
# 60000 images and labels
# Each image is 28x28 pixels
def load_dataset():
    images, labels = mndata.load_training()
    return images, labels

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

#images, labels = load_dataset()
#display_image(images, 235)
