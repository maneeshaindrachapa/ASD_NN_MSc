from kerastuner.tuners import Hyperband
import kerastuner as kt
import tensorflow as tf

from caps.capsnet import margin_loss
from info import participants, EEG_SHAPE
from tensorflow.keras import models as km
import numpy as np
import pandas as pd
from caps.capsnet.layers import ConvCaps2D, DenseCaps
from caps.capsnet.nn import squash, norm, mask_cid
from models import DenseBlock, TransitionBlock
import sys

INFO = 'Expected Arguments: [ MODEL_NAME ][convlstm , caps]'
REG = 'l1_l2'

# hyperparameter tuning for ConvLSTM
def build_model_convlstm(hp):
    input = tf.keras.layers.Input(shape=EEG_SHAPE)
    x = input
    timesteps = EEG_SHAPE[0]
    ch_rows = EEG_SHAPE[1]
    ch_cols = EEG_SHAPE[2]
    bands = EEG_SHAPE[3]
    x = tf.keras.layers.Reshape((timesteps, ch_rows, ch_cols, bands))(x)
    for i in range(hp.Int('conv_lstm_blocks', 1, 5, default=1)):
        filters = hp.Int('filters_' + str(i), 16, 256, step=16)
        activation = hp.Choice("activation", ["relu", "tanh", "softmax"])
        x = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=(4, 1), activation=activation,
                                       recurrent_activation='hard_sigmoid', padding='same',
                                       return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if hp.Choice('pooling', ['avg', 'max']) == 'max':
        x = tf.keras.layers.MaxPool3D(pool_size=(1, 3, 3), padding='same',
                                      data_format='channels_first')(x)
    else:
        x = tf.keras.layers.AvgPool3D(pool_size=(1, 3, 3), padding='same',
                                      data_format='channels_first')(x)
    for i in range(hp.Int('conv_lstm_blocks_2', 1, 5, default=1)):
        filters_ = hp.Int('filters_' + str(i), 16, 256, step=16)
        x = tf.keras.layers.ConvLSTM2D(filters=filters_, kernel_size=(4, 1), activation=activation,
                                       recurrent_activation='hard_sigmoid', padding='same',
                                       return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    dropout = hp.Float("dropout", 0, 0.4, step=0.1, default=0.1)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Flatten()(x)

    label = tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer=REG, name='l')(x)
    score = tf.keras.layers.Dense(1, kernel_regularizer=REG, name='s')(x)
    model = tf.keras.Model(inputs=input, outputs=[label, score])
    regular_loss = {'l': 'categorical_crossentropy', 's': 'mae'}
    metrics = {'l': 'acc'}
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss=regular_loss,
        metrics=metrics)

    return model


def build_model_caps(hp):
    input = tf.keras.layers.Input(shape=EEG_SHAPE)
    x = input
    timesteps = EEG_SHAPE[0]
    ch_rows = EEG_SHAPE[1]
    ch_cols = EEG_SHAPE[2]
    bands = EEG_SHAPE[3]
    x = tf.keras.layers.Reshape((timesteps, ch_rows* ch_cols, bands))(x)
    for i in range(hp.Int('conv_blocks', 1, 5, default=1)):
        filters = hp.Int('filters_' + str(i), 8, 256, step=16)
        activation = hp.Choice("activation", ["relu", "tanh", "softmax"])
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(4, 1), activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(2):
        x = DenseBlock(conv=4, filters=8, kernel_size=(4, 1))(x)
        x = TransitionBlock(filters=x.shape[-1])(x)
    x = ConvCaps2D(filters=8, filter_dims=4, kernel_size=(4, 1), strides=(2, 1))(x)
    x = tf.keras.layers.Lambda(squash)(x)
    x = DenseCaps(caps=2, caps_dims=8, routing_iter=3)(x)
    x = tf.keras.layers.Lambda(squash)(x)
    x = tf.keras.layers.Lambda(mask_cid)(x)

    label = tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer=REG, name='l')(x)
    score = tf.keras.layers.Dense(1, kernel_regularizer=REG, name='s')(x)
    model = tf.keras.Model(inputs=input, outputs=[label, score])
    capsule_loss = {'l': margin_loss, 's': 'mae'}
    metrics = {'l': 'acc'}
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss=capsule_loss,
        metrics=metrics)

    return model

if __name__ == '__main__':
    assert len(sys.argv) == 2, INFO
    model = sys.argv[1].strip().lower()
    if model=='convlstm':
        tuner_convlstm = kt.Hyperband(build_model_convlstm, objective=kt.Objective("val_l_acc", direction="max"),
                                  max_epochs=200, hyperband_iterations=2, directory="tuning",
                                  project_name='tuning-convlstm')
    if model=='caps':
        tuner_caps = kt.Hyperband(build_model_caps, objective=kt.Objective("val_l_acc", direction="max"),
                                  max_epochs=200, hyperband_iterations=2, directory="tuning",
                                  project_name='tuning-caps')

    # load dataset
    tf.random.set_seed(42)
    np.random.seed(42)
    FRACTION = 0.7
    print('loading EEG dataset...', end=' ', flush=True)
    eeg_dataset = np.load('data/data-processed-bands.npz')

    num_train = int(len(participants) * FRACTION)
    p_train = set(np.random.choice(participants, num_train, replace=False))
    print('OK')
    # features
    X_TRAIN, X_TEST = np.zeros((0, *EEG_SHAPE)), np.zeros((0, *EEG_SHAPE))  # type: np.ndarray # EEG features
    # labels
    Y_TRAIN, Y_TEST = np.zeros((0,)), np.zeros((0,))
    Z_TRAIN, Z_TEST = np.zeros((0,)), np.zeros((0,))
    for p in participants:
        _x = eeg_dataset[f'{p}_x']
        _y = np.full(len(_x), eeg_dataset[f'{p}_bc'])
        _z = np.full(len(_x), eeg_dataset[f'{p}_r'])
        if p in p_train:
            X_TRAIN = np.append(X_TRAIN, _x, axis=0)
            Y_TRAIN = np.append(Y_TRAIN, _y, axis=0)
            Z_TRAIN = np.append(Z_TRAIN, _z, axis=0)
        else:
            X_TEST = np.append(X_TEST, _x, axis=0)
            Y_TEST = np.append(Y_TEST, _y, axis=0)
            Z_TEST = np.append(Z_TEST, _z, axis=0)
    Y_TRAIN = tf.keras.utils.to_categorical(Y_TRAIN, num_classes=2)
    Y_TEST = tf.keras.utils.to_categorical(Y_TEST, num_classes=2)
    print(f'TRAINING: X={X_TRAIN.shape}, Y={Y_TRAIN.shape}, Z={Z_TRAIN.shape}')
    print(f'TESTING: X={X_TEST.shape}, Y={Y_TEST.shape}, Z={Z_TEST.shape}')
    D_TRAIN = [[X_TRAIN], [Y_TRAIN, Z_TRAIN]]
    D_TEST = [[X_TEST], [Y_TEST, Z_TEST]]

    # Hyperparameter tuning ConvLSTM
    if model == 'convlstm':
        tuner_convlstm.search(D_TRAIN[0], D_TRAIN[1],
                              validation_data=(D_TEST[0], D_TEST[1]),
                              epochs=30,
                              callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])
        best_model = tuner_convlstm.get_best_models(1)[0]
        best_hyperparameters = tuner_convlstm.get_best_hyperparameters(1)[0]

    # Hyperparameter tuning Caps
    if model == 'caps':
        tuner_caps.search(D_TRAIN[0], D_TRAIN[1],
                              validation_data=(D_TEST[0], D_TEST[1]),
                              epochs=30,
                              callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])
        best_model = tuner_caps.get_best_models(1)[0]
        best_hyperparameters = tuner_caps.get_best_hyperparameters(1)[0]

    print(best_hyperparameters)
