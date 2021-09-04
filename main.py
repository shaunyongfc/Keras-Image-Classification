# A copy of the jupyter notebook main.ipynb in executable python file

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from parameters import *
import image_loader
import keras_model

# Define category names
CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


class ImageClass():
    def __init__(self):
        """
        Main function that makes use of written functions in other files.
        """
        X_train, y_train = image_loader.get_images_train()
        X_val, y_val = image_loader.get_images_test()
        y_train = to_categorical(y_train, num_classes=6)
        y_val = to_categorical(y_val, num_classes=6)
        self.model = keras_model.create_model()
        self.history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BS)

    def image_predict(self, number):
        """
        Predict a category from a numbered file in the pred folder.
        """
        X_pred = image_loader.get_pred_image(number)
        y_pred = int(np.argmax(self.model.predict(X_pred), axis=-1))
        return CATEGORIES[y_pred]

    def save_model(self, name='image_class'):
        """
        Save the model into a folder of given path.
        """
        self.model.save(name)


if __name__ == '__main__':
    image_class = ImageClass()
    image_class.save_model()
