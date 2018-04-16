# Imports
import utils
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.models import Sequential

# Define training parameters
BATCH_SIZE  = 128
NUM_CLASSES = 10
EPOCHS      = 10

# 80% training set, 20% validation set
def split_dataset(images, labels):
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)
   # X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
   # X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
    #y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    #y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)
    return X_train, X_val, y_train, y_val

def reshape(X):
    return X.reshape(X.shape[0], 28, 28, 1)

def one_hot_encode(y):
    return keras.utils.to_categorical(y, NUM_CLASSES)

def get_next_batch(X, y):
    # Will contains images and labels
    X_batch = np.zeros((BATCH_SIZE, 28, 28, 1))
    y_batch = np.zeros(BATCH_SIZE)

    while True:
        for i in range(0, BATCH_SIZE):
            random_index = np.random.randint(len(X))
            X_batch[i] = X[random_index]
            y_batch[i] = y[random_index]
        print(X_batch.shape)
        print(y_batch.shape)
        yield X_batch, y_batch


def get_conv2d_model():
    model = Sequential()
    optimizer = Adam(lr=0.001)

    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(28, 28, 1)))
    model.add(Convolution2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer=optimizer, loss="mse")

    return model

def train(model, X_train, y_train, X_val, y_val):
    # Stop the training if delta val loss after 2 Epochs < 0.001
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=0, mode='auto')
    # Save the best model depending on the val_loss
    #model_checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

    model.fit_generator(
        generator=get_next_batch(X_train, y_train),
        samples_per_epoch=20,
        nb_epoch=EPOCHS,
        validation_data=get_next_batch(X_val, y_val),
        nb_val_samples=len(X_val)
        #callbacks=[early_stopping, model_checkpoint]
    )

    return model

# Load dataset
images, labels = utils.load_dataset()
images = np.array(images)
labels = np.array(labels)

# Prepare dataset
X_train, X_val, y_train, y_val = split_dataset(images, labels)
X_train = reshape(X_train)
X_val = reshape(X_val)
y_train = one_hot_encode(y_train)
y_val = one_hot_encode(y_val)

#print(X_train.shape)
#print(X_val.shape)

#print(y_train.shape)

# Get Conv2D model
model = get_conv2d_model()

# Train it
trained_model = train(model, X_train, y_train, X_val, y_val)


#utils.display_image(images, 234)
#utils.display_label(labels, 234)