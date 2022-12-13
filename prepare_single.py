import pandas as pd
from info import participants, epochs, source_cols


def readEEGData():
    # DataFrame to store all recordings
    df = pd.DataFrame(columns=['Participant', 'Epoch', *source_cols])

    # rename / reorganize columns and save to file
    for epoch in epochs:
        print(epoch)
        filename = f'data_1/eeg/005/005_{epoch}.csv'
        print(f'\t{filename}...')
        df_ = pd.read_csv(filename)
        df_['Participant'] = "004"
        df_['Epoch'] = epoch
        df = df.append(df_, ignore_index=True)


    print("Read Completed\n")
    print(df.head)

    print('Saving Data...')
    dest_filename = '02-data-original.ftr'
    df.to_feather(dest_filename)

if __name__ == '__main__':
    readEEGData()
