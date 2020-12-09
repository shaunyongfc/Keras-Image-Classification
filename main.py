import numpy as np
import pandas as pd
from keras.utils import to_categorical
from parameters import *
import image_loader
import keras_model

def main():
    training_images = image_loader.get_images_train()
    X_train = training_images['image']
    y_train = to_categorical(training_images['category'], num_classes=6)
    model = keras_model.create_model()
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BS)

if __name__ == '__main__':
    main()
