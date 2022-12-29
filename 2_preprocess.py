#!/usr/bin/env python3
import mne
import numpy as np
import pandas as pd

import info
from info import source_cols, target_cols, sampling_freq, measurement_unit

FILTER_DATA = True  # To pass data in bandpass filter


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


if __name__ == '__main__':
    preprocessData()
