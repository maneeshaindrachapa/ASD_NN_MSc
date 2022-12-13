import os
import sys
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as k
from sklearn.metrics import classification_report, r2_score, mean_absolute_error, explained_variance_score

import models
from info import participants, EEG_SHAPE, IRT_SHAPE
from caps.capsnet.losses import margin_loss

tf.random.set_seed(42)
np.random.seed(42)


# load EEG dataset
print('loading EEG dataset...', end=' ', flush=True)
eeg_dataset = np.load('002-data-processed-bands.npz')
print('OK')

X_TRAIN =np.zeros((0, *EEG_SHAPE))

_x = eeg_dataset[f'005_x']

X_TRAIN = np.append(X_TRAIN, _x, axis=0)
model = models.CONV(EEG_SHAPE)
model.load_weights("./weights/CONV.hdf5")
[label, score] = model.predict(X_TRAIN,  verbose=2)
label = (np.argmax(label, axis=1))
count_true = np.count_nonzero(label == 1)
count_false = np.count_nonzero(label == 0)
if count_true>count_false:
    print("true")
else:
    print("false")

