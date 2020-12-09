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
    image = Image.open(image_path)
    if category == '':
        return pd.Series([np.array(image)], index=['image'])
    else:
        return pd.Series([np.array(image), category], index=['image', 'category'])

def get_images(folder_path, category=''):
    path_list = glob.glob(os.path.join(folder_path, category, '*.jpg'))
    image_series = []
    for image_path in path_list:
        image_series.append(generate_entry(image_path, category))
    return pd.concat(image_series, axis=1).T
