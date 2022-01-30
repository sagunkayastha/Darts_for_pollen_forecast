
import os
import pandas as pd
import numpy as np
from itertools import chain


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
