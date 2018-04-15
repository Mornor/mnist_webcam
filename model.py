# Imports
import utils
import numpy as np
from sklearn.model_selection import train_test_split

# Convert images and labels from list to array and split the dataset
# 80% is training set
# 20% is validation set
def split_dataset(images, labels):
    images = np.array(images)
    labels = np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)
    return X_train, X_val, y_train, y_val



images, labels = utils.load_dataset()
X_train, X_val, y_train, y_val = split_dataset(images, labels)

#utils.display_image(images, 234)
#utils.display_label(labels, 234)