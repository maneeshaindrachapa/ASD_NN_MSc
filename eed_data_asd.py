import mne
import numpy as np
import pandas as pd
import pywt
from scipy.integrate import simps
import sys
from info import participants, epochs, NUM_BANDS, SLICE_SHAPE, SLICE_WINDOW, SLICE_STEP, SRC_FREQ, TARGET_FREQ, \
    SAMPLING_PERIOD, TARGET_BANDS, source_cols, target_cols, sampling_freq, measurement_unit
import info

READ_DATA = False

PREPROCESS_DATA = False
FILTER_DATA = True

GEN_DATA_SET = False
CREATE_FULL_DATASET = True  # create full data set preprocessed data + subjects data
CREATE_BANDS_DATASET = True  # create bands dataset using full dataset

TRAIN_TEST = False
INFO = 'Expected Arguments: [ R | P | G ] R-READ, P-PREPROCESS, G-GENERATE'

def readEEGData():
    # DataFrame to store all recordings
    df = pd.DataFrame(columns=['Participant', 'Epoch', *source_cols])

    # read all files into DataFrame
    print(f'Reading data from {len(participants)} participants...')

    # rename / reorganize columns and save to file
    for i, participant in enumerate(participants):
        for epoch in epochs:
            filename = f'data/eeg/{participant}/{participant}_{epoch}.csv'
            print(f'\t{filename}...')
            df_ = pd.read_csv(filename)
            df_['Participant'] = participant
            df_['Epoch'] = epoch
            df = df.append(df_, ignore_index=True)
        print(f'{i + 1} of {len(participants)} completed')
    print("Read Completed\n")
    print(df.head)
    print('Saving Data...')
    dest_filename = 'data/data-original.ftr'
    df.to_feather(dest_filename)


def preprocessData():
    # read original dataset
    print('Reading dataset...', sep=' ', flush=True)
    df = pd.read_feather('data/data-original.ftr')
    print('OK')

    # if any new columns need to be created, create them beforehand
    print('Creating new columns...', sep=' ', flush=True)
    new_cols = set(target_cols).difference(source_cols)
    # Replace newly created column values with NaN
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
        data = mne.io.RawArray(data=_in.to_numpy().transpose() * measurement_unit, info=_info)
        data.set_montage('standard_1020')
        data.interpolate_bads(reset_bads=True)
        if FILTER_DATA:
            data = data.filter(l_freq=1, h_freq=50)
        # append to output
        _out = data.to_data_frame().rename(columns={'time': 'T'})  # type: pd.DataFrame
        _out['Participant'] = i[0]
        _out['Epoch'] = i[1]
        _out = _out.set_index(['Participant', 'Epoch', 'T'])[target_cols]
        df_out = df_out.append(_out)
    if FILTER_DATA:
        vizualizeElectrogramEEG("figures/filtered-eeg", df_out)
    else:
        vizualizeElectrogramEEG("figures/un-filtered-eeg", df_out)

    print(df_out.head)
    print(len(df_out.index))
    print('OK')
    print('Saving data')
    df_out.reset_index().to_feather('data/data-clean.ftr')
    print('OK')


def vizualizeElectrogramEEG(file_name: str, df: pd.DataFrame):
    _cols = df.columns.to_list()  # type: list
    _info = mne.create_info(ch_names=_cols, sfreq=sampling_freq, ch_types='eeg')  # type: dict
    data = mne.io.RawArray(data=df.to_numpy().transpose() * measurement_unit, info=_info)
    data.set_montage('standard_1020')
    ica = mne.preprocessing.ICA(n_components=len(info.target_cols), random_state=97, max_iter=800)
    ica.fit(data)
    ica.plot_sources(data, show_scrollbars=False)
    fig1 = ica.plot_components()

    for idx, fig_ in enumerate(fig1):
        fname = '{}'.format(idx)
        fig_.savefig(file_name + "ICA" + fname, bbox_inches='tight')

    # plot an overlay of the original signal against the reconstructed signal with the artifactual ICs excluded
    ica.plot_overlay(data, exclude=[0], picks='eeg')
    # plot some diagnostics of each IC
    ica.plot_properties(data, picks=[0])

    fig = data.plot(start=20, duration=5, n_channels=len(info.target_cols))
    fig.savefig(file_name, bbox_inches='tight')


def scale(_x: np.array):
    return (_x - _x.min()) / (_x.max() - _x.min())


def genDataSet():
    print('Loading Data')
    data = pd.read_feather('data/data-clean.ftr')
    print('OK')
    print('Loading Labels')
    labels = pd.read_csv('data/SUBJECTS.csv', dtype={'ID': object}).set_index('ID')
    bc_col = 'ASD'
    cc_col = 'EEG'
    r_col = 'ADOS2'
    print('OK')

    BANDS = np.arange(NUM_BANDS) + 1  # frequencies (1 Hz - 50 Hz) range

    # define dict to store output
    dataset = {}
    # wavelet transform properties
    wavelet = 'cmor1.5-1.0'  # complex morlet wavelet (Bandwidth - 1.5 Hz, Center Frequency - 1.0 Hz)
    # wavelet = 'sym9'  # symlet 9 wavelet
    scales = SRC_FREQ / BANDS  # scales corresponding to frequency bands

    if CREATE_FULL_DATASET:
        # generate values for x, y
        print('Generating X, Y')
        data = data.set_index('Participant')
        for i, p in enumerate(participants):
            print(f'Participant: {p} - ', flush=True, end='')

            bc = labels.loc[p][bc_col]
            cc = labels.loc[p][cc_col]
            r = labels.loc[p][r_col]
            dp = data.loc[p].set_index('Epoch')
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
            dataset[f'{p}_x'] = p_data
            dataset[f'{p}_bc'] = bc
            dataset[f'{p}_cc'] = cc
            dataset[f'{p}_r'] = r
            print(p_data.shape)
        print('OK')

        # save dataset
        print('Saving processed data')
        np.savez_compressed('data/data-processed.npz', **dataset)
        print('OK')

    if CREATE_BANDS_DATASET:
        # extract delta, theta, alpha, beta, and gamma frequency bands
        print('Reducing to frequency bands')
        dataset = np.load('data/data-processed.npz')
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
        np.savez_compressed('data/data-processed-bands.npz', **band_dataset)
        print('OK')


if __name__ == '__main__':
    assert len(sys.argv) == 2, INFO
    mode = sys.argv[1].strip().upper()
    if mode=='R': READ_DATA=True
    if mode=='P': PREPROCESS_DATA=True
    if mode=='G': GEN_DATA_SET=True
    if READ_DATA:
        print("====READ DATA====")
        readEEGData()
    if PREPROCESS_DATA:
        print("====PREPROCESS DATA====")
        preprocessData()
    if GEN_DATA_SET:
        print("====GENERATE DATA====")
        genDataSet()
