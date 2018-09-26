# Imports
import utils
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adadelta, Adam
from keras.models import Sequential

# Define training parameters
#BATCH_SIZE  = 2000
BATCH_SIZE  = 128
NUM_CLASSES = 10 # (0 to 9)
EPOCHS      = 5

# 80% training set, 20% validation set
def split_dataset(images, labels):
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)
    return X_train, X_val, y_train, y_val

def reshape(X):
    return X.reshape(X.shape[0], 28, 28, 1)

def get_next_batch(X, y):
    # Will contains images and labels
    X_batch = np.zeros((BATCH_SIZE, 28, 28, 1))
    y_batch = np.zeros((BATCH_SIZE, 10))

    while True:
        for i in range(0, BATCH_SIZE):
            random_index = np.random.randint(len(X))
            X_batch[i] = X[random_index]
            y_batch[i] = y[random_index]
        yield X_batch, y_batch

def get_conv2d_model():
    model = Sequential()
    optimizer = Adam()

    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(28, 28, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    #model.summary()

    return model

def train_with_generator(model, X_train, y_train, X_val, y_val):
    # Stop the training if delta val loss after 2 Epochs < 0.001
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=0, mode='auto')
    # Save the best model depending on the val_loss
    #model_checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

    model.fit_generator(
        generator=get_next_batch(X_train, y_train),
        steps_per_epoch=1000,
        epochs=EPOCHS,
        validation_data=get_next_batch(X_val, y_val),
        #validation_steps=len(X_val/2),
        validation_steps=1000
        #callbacks=[early_stopping, model_checkpoint]
    )

    return model

# Load dataset
(X_train, y_train), (X_test, y_test) = utils.load_dataset()

# Prepare dataset
X_train, X_val, y_train, y_val = split_dataset(X_train, y_train)
X_train = reshape(X_train)
X_val = reshape(X_val)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# Get Conv2D model
model = get_conv2d_model()

# Train it
#trained_model = train_with_generator(model, X_train, y_train, X_val, y_val)
trained_model = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=64)
model.save('test_model.model')

# Check the validity of the model on test dataset
#X_test = reshape(X_test)
#y_test = to_categorical(y_test)
#val_loss, val_acc = trained_model.evaluate(X_test, y_test)
#print(val_loss)
#print(val_acc)

# Save it in case of trained with fit_generator()
#utils.save_model(trained_model, 'test_model')
