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
    if category == '':
        return pd.Series([np.asarray(image) / 255.0], index=['image'])
    else:
        return pd.Series([np.asarray(image) / 255.0, CATEGORIES.index(category)], index=['image', 'category'])

def get_images(folder_path, category=''):
    """
    Given folder path, return a pandas dataframe of image data.
    """
    path_list = glob.glob(os.path.join(folder_path, category, '*.jpg'))
    image_series = []
    for image_path in path_list:
        image_series.append(generate_entry(image_path, category))
    return pd.concat(image_series, axis=1).T

def get_images_train():
    """
    Get all image data of the training set.
    """
    all_images = pd.concat([get_images(PATH_TRAIN, a) for a in CATEGORIES])
    return all_images.sample(frac=1)

def get_images_test():
    """
    Get all image data of the test set.
    """
    all_images = pd.concat([get_images(PATH_TEST, a) for a in CATEGORIES])
    return all_images.sample(frac=1)
