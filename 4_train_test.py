import os
import sys
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, r2_score, mean_absolute_error, explained_variance_score, \
    confusion_matrix,mean_squared_error
from tensorflow import keras as k

import models
from caps.capsnet.losses import margin_loss
from info import participants, EEG_SHAPE, IRT_SHAPE
import time

start_time = time.time()
tf.random.set_seed(42)
np.random.seed(42)
INFO = 'Expected Arguments: [OPTIONAL] [ train | test | info ] [ MODEL_NAME ]'
FRACTION = 0.7


def saveTrainTestLostFig(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'figures/train-test-lost-epoch-{model_name}.png', bbox_inches='tight')


def plot_history(history):
    """
    Plot the training history
    """
    metrics = 'loss'
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
    plt.savefig(f'figures/train-history-{model_name}.png', bbox_inches='tight')


def ConfusionMatrix(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0: 0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.savefig(f'figures/confusion-matrix-{model_name}.png', bbox_inches='tight')


if __name__ == '__main__':
    # parse command line arguments
    assert len(sys.argv) == 3, INFO
    mode = sys.argv[1].strip().lower()
    model_name = sys.argv[2].strip().lower()
    assert mode in ['train', 'test', 'info'], INFO
    assert model_name in ['conv', 'convlstm', 'bilstm', 'lstm', 'caps', 'gru'], INFO
    training = mode == 'train'
    testing = mode == 'test'
    info = mode == 'info'

    # load EEG dataset
    print('loading EEG dataset...', end=' ', flush=True)
    eeg_dataset = np.load('data_1/data-processed-bands.npz')
    print('OK')

    # train-test-split on participant ID
    print('performing train-test split...', end=' ', flush=True)
    num_train = int(len(participants) * FRACTION)
    p_train = set(np.random.choice(participants, num_train, replace=False))
    print('OK')

    # create test and train data
    # features
    X_TRAIN, X_TEST = np.zeros((0, *EEG_SHAPE)), np.zeros((0, *EEG_SHAPE))  # type: np.ndarray # EEG features
    # labels
    Y_TRAIN, Y_TEST = np.zeros((0,)), np.zeros((0,))  # type: np.ndarray # prediction label
    Z_TRAIN, Z_TEST = np.zeros((0,)), np.zeros((0,))  # type: np.ndarray # ados-2 score
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
    # one-hot encode class label
    Y_TRAIN = k.utils.to_categorical(Y_TRAIN, num_classes=2)
    Y_TEST = k.utils.to_categorical(Y_TEST, num_classes=2)
    print(f'TRAINING: X={X_TRAIN.shape}, Y={Y_TRAIN.shape}, Z={Z_TRAIN.shape}')
    print(f'TESTING: X={X_TEST.shape}, Y={Y_TEST.shape}, Z={Z_TEST.shape}')
    print('OK')

    print('Creating Variables...', end=' ', flush=True)
    # global variables
    regular_loss = {'l': 'categorical_crossentropy', 's': 'mae'}
    capsule_loss = {'l': margin_loss, 's': 'mae'}
    metrics = {'l': 'acc'}
    # model-specific configurations
    losses_dict = {
        'conv': regular_loss,
        'lstm': regular_loss,
        'bilstm': regular_loss,
        'convlstm': regular_loss,
        'caps': capsule_loss,
        'gru': regular_loss,
    }
    shapes_dict = {
        'conv': [EEG_SHAPE],
        'lstm': [EEG_SHAPE],
        'bilstm': [EEG_SHAPE],
        'convlstm': [EEG_SHAPE],
        'caps': [EEG_SHAPE],
        'gru': [EEG_SHAPE],
    }
    models_dict = {
        'conv': models.CONV,
        'lstm': models.LSTM,
        'bilstm': models.BILSTM,
        'convlstm': models.ConvLSTM,
        'caps': models.CAPS,
        'gru': models.GRU_,
    }
    # model-specific input data
    D_TRAIN: List = ...
    D_TEST: List = ...
    if model_name in ['conv', 'convlstm', 'bilstm', 'lstm', 'caps', 'gru']:
        D_TRAIN = [[X_TRAIN], [Y_TRAIN, Z_TRAIN]]
        D_TEST = [[X_TEST], [Y_TEST, Z_TEST]]
    print('OK')

    print('Training and Evaluation')
    # get model configuration
    model_loss = losses_dict[model_name]
    loss_weights = [1, 0.05]
    input_shape = shapes_dict[model_name]
    model = models_dict[model_name](*input_shape)
    optimizer = k.optimizers.Adam(0.0005)
    save_path = f'weights/{model.name}.hdf5'
    # build model
    model.compile(optimizer=optimizer, loss=model_loss, loss_weights=loss_weights, metrics=metrics)
    # model information
    if info:
        model.summary(line_length=150)
    # training phase
    if training:
        # load pre-trained weights when available
        if os.path.exists(save_path):
            model.load_weights(save_path)
        # train
        save_best = k.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True,
                                                save_weights_only=True, verbose=0)
        history = model.fit(D_TRAIN[0], D_TRAIN[1], batch_size=32, epochs=500, validation_data=(D_TEST[0], D_TEST[1]),
                            callbacks=[save_best], verbose=2)
        saveTrainTestLostFig(history)
        plot_history(history)
    if testing:
        model.load_weights(save_path)
        [label, score] = model.predict(D_TEST[0], batch_size=32, verbose=2)
        y_pred = np.argmax(label, axis=1)
        y_true = np.argmax(D_TEST[1][0], axis=1)
        # Save confusion matrix
        ConfusionMatrix(y_true, y_pred)
        print("Classification Task")
        print(classification_report(y_true, y_pred))
        print("Regression Task")
        r_true = D_TEST[1][1]
        r_pred = score
        print(f'R^2 = {r2_score(r_true, r_pred)}')
        print(f'MAE = {mean_absolute_error(r_true, r_pred)}')
        print(f'RMSE = {(mean_squared_error(r_true, r_pred))**0.5}')
        print(f'EVS = {explained_variance_score(r_true, r_pred)}')
        print("\nModel:"+model_name.upper())
        print("--- %s seconds ---" % (time.time() - start_time))
    print('Done')
