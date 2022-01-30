
import os
import pandas as pd
import numpy as np
from itertools import chain
from filling import det_regression



def fast_flatten(input_list):
    return list(chain.from_iterable(input_list))


def prep_dataframe(folder, category):
    var = ['DeviceId', 'Latitude', 'Longitude', 'SolarIrradiance', 'ds',
           'OrdinalDay', 'HourOfDay', 'ActualPPM3', 'CategoryCode',
           'DailyHistory', 'HourlyHistory', 'humi',
           'pop12', 'qpf', 'temp', 'wspd']
    frames = list()
    for csv in [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.endswith('.csv')]:
        tmp_ = pd.read_csv(csv)
        if list(tmp_.columns) == var:
            frames.append(tmp_)

    COLUMN_NAMES = frames[0].columns
    df_dict = dict.fromkeys(COLUMN_NAMES, [])

    for col in COLUMN_NAMES:
        extracted = (frame[col] for frame in frames)

        # Flatten and save to df_dict
        df_dict[col] = fast_flatten(extracted)

    df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]

    df['humi'] = df['humi'].apply(lambda x: x if x < 9999 else np.NaN)
    df['pop12'] = df['pop12'].apply(lambda x: x if x < 9999 else np.NaN)
    df['qpf'] = df['qpf'].apply(lambda x: x if x < 9999 else np.NaN)
    df['temp'] = df['temp'].apply(lambda x: x if x < 9999 else np.NaN)
    df['wspd'] = df['wspd'].apply(lambda x: x if x < 9999 else np.NaN)

    df2 = df[df['CategoryCode'] == category].astype('object')
    return df2.reset_index(drop=True)


def filling_cleaning(dfx):

    gps = dfx.groupby(['DeviceId']).size()
    lengths = pd.DataFrame(gps, columns=['Len']).sort_values(
            by='Len', ascending=False)

    # for did in lengths.index[1]:

    did = 99057

    df = dfx[dfx.DeviceId == did].sort_values(
                    by=['Latitude', 'Longitude', 'OrdinalDay', 'HourOfDay']
                    ).reset_index(drop=True)

    df = df[['ds', 'ActualPPM3', 'SolarIrradiance',
            'humi', 'pop12', 'qpf', 'temp', 'wspd']]
    df['ds'] = df['ds'].apply(lambda x: x.split(' ')[0])
    cols = ['ActualPPM3', 'SolarIrradiance',
            'humi', 'pop12', 'qpf', 'temp', 'wspd']
    df[cols] = df[cols].apply(pd.to_numeric)
    gps = df.groupby(['ds']).mean()

    idx = pd.date_range('01-01-2020', '12-31-2021')
    gps.index = pd.DatetimeIndex(gps.index)

    gps = gps.reindex(idx, fill_value=np.NaN)
    gps['time'] = pd.to_datetime(gps.index)
    df_avg = gps.reset_index(drop=True)

    df_avg_filled, d_r = det_regression(df_avg[cols])
    df_avg_filled['time'] = pd.to_datetime(gps.index)

    return df_avg_filled

def test_filling_cleaning(dfx):
    gps = dfx.groupby(['DeviceId']).size()
    lengths = pd.DataFrame(gps, columns=['Len']).sort_values(
            by='Len', ascending=False)

    # for did in lengths.index[1]:

    did = 99057

    df = dfx[dfx.DeviceId == did].sort_values(
                    by=['Latitude', 'Longitude', 'OrdinalDay', 'HourOfDay']
                    ).reset_index(drop=True)

    df = df[['ds', 'ActualPPM3', 'SolarIrradiance',
            'humi', 'pop12', 'qpf', 'temp', 'wspd']]
    df['ds'] = df['ds'].apply(lambda x: x.split(' ')[0])
    cols = ['ActualPPM3', 'SolarIrradiance',
            'humi', 'pop12', 'qpf', 'temp', 'wspd']
    df[cols] = df[cols].apply(pd.to_numeric)
    gps = df.groupby(['ds']).mean()

    idx = pd.date_range('01-01-2022', '01-15-2022')
    gps.index = pd.DatetimeIndex(gps.index)

    gps = gps.reindex(idx, fill_value=np.NaN)
    gps['time'] = pd.to_datetime(gps.index)
    df_avg = gps.reset_index(drop=True)

    df_avg_filled, d_r = det_regression(df_avg[cols])
    df_avg_filled['time'] = pd.to_datetime(gps.index)

    return df_avg_filled