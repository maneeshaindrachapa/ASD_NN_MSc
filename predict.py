import mne
import pandas as pd
import numpy as np
import pywt
from scipy.integrate import simps
import models
from info import source_cols, target_cols, sampling_freq, epochs, NUM_BANDS, SLICE_SHAPE, \
    SLICE_WINDOW, SLICE_STEP, SRC_FREQ, TARGET_FREQ, \
    SAMPLING_PERIOD, TARGET_BANDS
from tensorflow import keras as k
from info import EEG_SHAPE
from capsnet import margin_loss
import time

FILTER_DATA = True  # To pass data in bandpass filter
global READ_COMPLETE, PREPROCESS_COMPLETE, GEN_BANDS_COMPLETE
READ_COMPLETE = False
PREPROCESS_COMPLETE = False
GEN_BANDS_COMPLETE = False


def scale(_x: np.array):
    return (_x - _x.min()) / (_x.max() - _x.min())


def readEEGData(participant_id):
    # DataFrame to store all recordings
    df = pd.DataFrame(columns=['Participant', 'Epoch', *source_cols])
    # rename / reorganize columns and save to file
    for epoch in epochs:
        filename = f'./Files/{participant_id}_{epoch}.csv'
        print(f'\t{filename}...')
        df_ = pd.read_csv(filename)
        df_['Participant'] = participant_id
        df_['Epoch'] = epoch
        df = df.append(df_, ignore_index=True)
    print("Read Completed\n")

    print('Saving Data...')
    dest_filename = f'./Files/{participant_id}-data-original.ftr'
    df.to_feather(dest_filename)
    global READ_COMPLETE
    READ_COMPLETE = True


def preprocessData(participant_id):
    # read original dataset
    print('Reading dataset...', sep=' ', flush=True)
    df = pd.read_feather(f'./Files/{participant_id}-data-original.ftr')
    print('OK')
    # if any new columns need to be created, create them beforehand
    print('Creating new columns...', sep=' ', flush=True)
    new_cols = set(target_cols).difference(source_cols)
    for col in new_cols:
        df[col] = np.nan
    print('OK')
    # index by participant, epoch and time
    print('Updating Index')
    df = df.set_index(['Participant', 'Epoch']).sort_index()
    print('OK')
    # DataFrame to store pre-processed recordings
    df_out = pd.DataFrame(columns=target_cols, index=pd.MultiIndex.from_arrays(arrays=[[], [], []],
                                                                               names=['Participant', 'Epoch', 'T']))
    print('Interpolating missing columns')
    for i in df.index.unique():
        # select data slice
        _in = df.loc[i]  # type: pd.DataFrame
        # define (all) columns
        _cols = _in.columns.to_list()  # type: list
        # define (bad) columns
        _bads = _in.columns[_in.isna().any()].tolist()  # type: list
        # interpolate bad columns (if any)
        _info = mne.create_info(ch_names=_cols, sfreq=sampling_freq, ch_types='eeg')  # type: dict
        _info['bads'] = _bads
        data = mne.io.RawArray(data=np.array(_in, dtype=float).transpose() * sampling_freq, info=_info)
        data.set_montage('standard_1020')
        data.interpolate_bads(reset_bads=True)
        if FILTER_DATA:
            data = data.filter(l_freq=1, h_freq=35)
        # append to output
        _out = data.to_data_frame().rename(columns={'time': 'T'})  # type: pd.DataFrame
        _out['Participant'] = i[0]
        _out['Epoch'] = i[1]
        _out = _out.set_index(['Participant', 'Epoch', 'T'])[target_cols]
        df_out = df_out.append(_out)
    print('OK')
    print('Saving data')
    df_out.reset_index().to_feather(f'./Files/{participant_id}-data-clean.ftr')
    print('OK')
    global PREPROCESS_COMPLETE
    PREPROCESS_COMPLETE = True


def generateBands(participant_id):
    print('Loading Data')
    data = pd.read_feather(f'./Files/{participant_id}-data-clean.ftr')
    print('OK')
    print('Loading Labels')
    print('OK')

    BANDS = np.arange(NUM_BANDS) + 1  # frequencies (1 Hz - 50 Hz)
    CREATE_FULL_DATASET = True
    CREATE_BANDS_DATASET = True
    dataset = {}
    wavelet = 'cmor1.5-1.0'  # complex morlet wavelet (Bandwidth - 1.5 Hz, Center Frequency - 1.0 Hz)
    scales = SRC_FREQ / BANDS  # scales corresponding to frequency bands
    if CREATE_FULL_DATASET:
        # generate values
        data = data.set_index('Participant')
        dp = data.loc[participant_id].set_index('Epoch')
        p_data = np.zeros((0, *SLICE_SHAPE))  # type: np.ndarray
        for j, e in enumerate(epochs[1:]):
            print(f'{e} ', flush=True, end='')
            de = dp.loc[e].set_index('T').to_numpy()  # type: np.ndarray # shape:(timestep, channel)
            # powers of each channel
            ch_p = []
            for ch in de.T:
                # find wavelet transform coefficients of channel signal
                c, _ = pywt.cwt(data=ch, scales=scales, wavelet=wavelet,
                                sampling_period=SAMPLING_PERIOD)  # type: np.ndarray
                # calculate abs square of c to obtain wavelet power spectrum
                ps = np.abs(c) ** 2  # type: np.ndarray
                # truncate p to avoid partial slices
                last_t = len(ch) // SRC_FREQ
                last_t -= (last_t - SLICE_WINDOW) % SLICE_STEP
                timesteps = last_t * SRC_FREQ
                l_trim = (len(ch) - timesteps) // 2
                ps = ps[:, l_trim:l_trim + timesteps]
                # down-scale the power spectrum to target frequency (helps to reduce kernel size later)
                E = SRC_FREQ // TARGET_FREQ
                ps = np.mean(ps.reshape((ps.shape[0], ps.shape[1] // E, E)), axis=-1)
                # append power of channel to array
                ch_p.append(ps.T)  # shape: (timestep, band)
            # stack each power spectrum
            ps = np.stack(ch_p, axis=1)  # shape: (timestep, channel, band)
            # chunk power spectrum into N slices of SLICE_SHAPE
            W = SLICE_WINDOW * TARGET_FREQ
            S = SLICE_STEP * TARGET_FREQ
            N = (len(ps) - W) // S
            ws = [ps[k * S:k * S + W].reshape(SLICE_SHAPE) for k in range(N)]
            # generate training data samples
            ds = np.stack(ws, axis=0)  # shape: (sample, timestep, row, col, band)
            # append data samples to participant data
            p_data = np.append(p_data, ds, axis=0)
        # add participant's data to output
        dataset[f'{participant_id}_x'] = p_data
        print(p_data.shape)
    print('OK')
    print('Saving processed data')
    np.savez_compressed(f'./Files/{participant_id}-data-processed.npz', **dataset)
    print('OK')

    if CREATE_BANDS_DATASET:
        # extract delta, theta, alpha, beta, and gamma frequency bands
        print('Reducing to frequency bands')
        dataset = np.load(f'./Files/{participant_id}-data-processed.npz')
        band_dataset = {}
        for key in dataset.keys():
            if key[-1] != 'x':
                band_dataset[key] = dataset[key]
                continue
            # power spectrum
            _ps = dataset[key]
            # band power (N x 30 x 5 x 10 x 5)
            _band_power = np.stack([simps(_ps[..., lo - 1:hi], axis=-1) for lo, hi in TARGET_BANDS],
                                   axis=-1)  # type: np.ndarray
            # differential entropy (DE) (N x 30 x 5 x 10 x 5)
            _de = np.log(_band_power)
            band_dataset[key] = _de
        print('OK')
        print('Saving frequency band data')
        np.savez_compressed(f'./Files/{participant_id}-data-processed-bands.npz',
                            **band_dataset)
        print('OK')
        global GEN_BANDS_COMPLETE
        GEN_BANDS_COMPLETE = True


def _predict(participant_id, model_name):
    start_time = time.time()
    print('loading EEG dataset...', end=' ', flush=True)
    eeg_dataset = np.load(f'./Files/data-processed-bands.npz')
    print('OK')

    regular_loss = {'l': 'categorical_crossentropy', 's': 'mae'}
    capsule_loss = {'l': margin_loss, 's': 'mae'}
    metrics = {'l': 'acc'}
    X_PRED = np.zeros((0, *EEG_SHAPE))
    _x = eeg_dataset[f'{participant_id}_x']

    X_PRED = np.append(X_PRED, _x, axis=0)
    if model_name == 'CONV':
        model = models.CONV(EEG_SHAPE)
        model_loss = regular_loss
    elif model_name == 'LSTM':
        model = models.LSTM(EEG_SHAPE)
        model_loss = regular_loss
    elif model_name == 'BILSTM':
        model = models.BILSTM(EEG_SHAPE)
        model_loss = regular_loss
    elif model_name == 'CONVLSTM':
        model = models.ConvLSTM(EEG_SHAPE)
        model_loss = regular_loss
    elif model_name == 'CAPS':
        model = models.CAPS(EEG_SHAPE)
        model_loss = capsule_loss
    elif model_name == 'GRU':
        model = models.GRU_(EEG_SHAPE)
        model_loss = regular_loss
    optimizer = k.optimizers.Adam(0.0005)
    loss_weights = [1, 0.05]
    model.compile(optimizer=optimizer, loss=model_loss, loss_weights=loss_weights, metrics=metrics)
    model.load_weights(f'weights/{model_name}.hdf5')

    [label, score] = model.predict(X_PRED, batch_size=32, verbose=2)
    print(label)
    percentage = np.sum(label,axis=0)
    total = np.sum(percentage)
    positive = (percentage[1]/total)*100
    negative = (percentage[0]/total)*100
    label = (np.argmax(label, axis=1))
    count_true = np.count_nonzero(label == 1)
    count_false = np.count_nonzero(label == 0)
    print("--------\nModel:"+model_name.upper())
    print("--- %s seconds ---" % (time.time() - start_time))
    print(percentage/30)
    predictions = [negative,positive]
    if count_true > count_false:
        predictions.append(1)
    else:
        predictions.append(0)
    return predictions


def predictStart(participant_id, model_name):
    readEEGData(participant_id)
    if (READ_COMPLETE):
        preprocessData(participant_id)
    if (PREPROCESS_COMPLETE):
        generateBands(participant_id)
    if (GEN_BANDS_COMPLETE):
        predict = _predict(participant_id, model_name.upper())
        return predict
