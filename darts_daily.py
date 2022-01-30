import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import TCNModel, RNNModel, BlockRNNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, r2_score
from darts.utils.missing_values import fill_missing_values
from darts.datasets import AirPassengersDataset, SunspotsDataset, EnergyDataset
from sklearn.preprocessing import MinMaxScaler

import joblib
from data_prep import prep_dataframe
from filling import det_regression

category = 'POL'
df1 = prep_dataframe('new_data/2020', category)
df2 = prep_dataframe('new_data/2021', category)

dfx = pd.concat([df1, df2])

gps = dfx.groupby(['DeviceId']).size()
lengths = pd.DataFrame(gps, columns=['Len']).sort_values(
        by='Len', ascending=False)

# for did in lengths.index[1]:

did = lengths.index[1]

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

scaler = MinMaxScaler()
df_avg_filled[['ActualPPM3', 'SolarIrradiance', 'humi',
               'pop12', 'qpf', 'temp', 'wspd']] = scaler.fit_transform(
                df_avg_filled[['ActualPPM3', 'SolarIrradiance', 'humi',
                'pop12', 'qpf', 'temp', 'wspd']])

# save_scaler
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)

x_df = df_avg_filled[['ActualPPM3', 'time']]
x = TimeSeries.from_dataframe(x_df, time_col='time')
x_covar = TimeSeries.from_dataframe(df_avg_filled.loc[:, df_avg_filled.columns != 'ActualPPM3'],  time_col='time')

Rnn_Model = RNNModel(input_chunk_length=30,
                     output_chunk_length=10,
                     n_rnn_layers=2)

Rnn_Model.fit(x,
              future_covariates=x_covar,
              epochs=100,
              verbose=True)

Rnn_Model.save_model('model.pth.tar')

