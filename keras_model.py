import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam
from parameters import *

def create_model():
    """
    Create the neural network model.
    """
    inputs = Input(shape=INPUT_SHAPE)
    # First convolutional layer
    x = Conv2D(16, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    # Second convolutional layer
    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    # Third convolutional layer
    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    # First fully connected layer
    x = Flatten()(x)
    x = Dense(100)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # Final fully connected layer
    x = Dense(6)(x)
    x = Activation("softmax")(x)
    model = Model(inputs=inputs, outputs=x, name="image_classification")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
