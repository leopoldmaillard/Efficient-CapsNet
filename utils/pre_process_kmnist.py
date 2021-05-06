import numpy as np
import tensorflow as tf
import os
import cv2
from tqdm.notebook import tqdm
tf2 = tf.compat.v2

# constants
KMNIST_IMG_SIZE = 28
KMNIST_TRAIN_IMAGE_COUNT = 60000
KMNIST_TEST_IMAGE_COUNT = 10000
PARALLEL_INPUT_CALLS = 16

def pre_process_train(ds):
    X = np.empty((KMNIST_TRAIN_IMAGE_COUNT, KMNIST_IMG_SIZE, KMNIST_IMG_SIZE, 1))
    y = np.empty((KMNIST_TRAIN_IMAGE_COUNT,))
        
    for index, d in tqdm(enumerate(ds.batch(1))):
        X[index, :, :] = d['image']
        y[index] = d['label']
    return X, y

def pre_process_test(ds):
    X = np.empty((KMNIST_TEST_IMAGE_COUNT, KMNIST_IMG_SIZE, KMNIST_IMG_SIZE, 1))
    y = np.empty((KMNIST_TEST_IMAGE_COUNT,))
        
    for index, d in tqdm(enumerate(ds.batch(1))):
        X[index, :, :] = d['image']
        y[index] = d['label']
    return X, y

