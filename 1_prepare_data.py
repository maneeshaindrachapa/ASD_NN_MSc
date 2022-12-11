import pandas as pd
from info import participants, epochs, source_cols

def readEEGData():
    # DataFrame to store all recordings
    df = pd.DataFrame(columns=['Participant', 'Epoch', *source_cols])

    # read all files into DataFrame
    print(f'Reading data from {len(participants)} participants...')

    # rename / reorganize columns and save to file
    for i, participant in enumerate(participants):
        for epoch in epochs:
            filename = f'data_1/eeg/{participant}/{participant}_{epoch}.csv'
            print(f'\t{filename}...')
            df_ = pd.read_csv(filename)
            df_['Participant'] = participant
            df_['Epoch'] = epoch
            df = df.append(df_, ignore_index=True)
        print(f'{i + 1} of {len(participants)} completed')
    print("Read Completed\n")
    print(df.head)

    print('Saving Data...')
    dest_filename = 'data_1/data-original.ftr'
    df.to_feather(dest_filename)

if __name__ == '__main__':
    readEEGData()