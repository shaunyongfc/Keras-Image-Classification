import numpy as np
import pandas as pd
import os
from os.path import join
import glob
from PIL import Image

CATEGORIES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
PATH_PRED = join(os.getcwd(), 'dataset/seg_pred/seg_pred/')
PATH_TEST = join(os.getcwd(), 'dataset/seg_test/seg_test/')
PATH_TRAIN = join(os.getcwd(), 'dataset/seg_train/seg_train/')

def generate_entry(image_path, category):
    """
    Given file path and of an image, return a pandas series of a numpy array of
    pixel data (width, height, channels), along with category in integer indices
    if category (string) is given.
    """
    image = Image.open(image_path)
    image = image.resize((150, 150))
    return np.array(image) / 255.0, CATEGORIES.index(category)

def get_images(folder_path):
    """
    Given folder path, return a pandas dataframe of image data.
    """
    image_X = []
    image_y = []
    for cat in CATEGORIES:
        path_list = glob.glob(os.path.join(folder_path, cat, '*.jpg'))
        for image_path in path_list:
            image_array, category_int = generate_entry(image_path, cat)
            image_X.append(image_array)
            image_y.append(category_int)
    X = np.stack(image_X)
    y = np.array(image_y)
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    return X, y

def get_images_train():
    """
    Get all image data of the training set.
    """
    return get_images(PATH_TRAIN)

def get_images_test():
    """
    Get all image data of the test set.
    """
    return get_images(PATH_TEST)

def get_pred_image(number):
    """
    Get image data from a numbered file in the pred folder.
    """
    image_path = os.path.join(PATH_PRED, str(number) + '.jpg')
    image = Image.open(image_path)
    image = image.resize((150, 150))
    return np.expand_dims((np.array(image) / 255.0), axis=0)
