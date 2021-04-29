# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:23:11 2021

@author: usuario
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from anomalias import samples
from sklearn.model_selection import train_test_split
from anomalias.mtsad_vae import MTSAD_CNN_VAE 


train_data, labels, scaler, columns_names, timestamp, length_data = samples.get_data(seq_length=60,
                                                                    shift_sample=60,
                                                                    aggregate_sec=1,
                                                                    drop=[X_train, X_val, y_train, y_val = train_test_split(
    train_data, labels, test_size=0.1, shuffle=False, random_state=42)],
                                                                    path='data/train/SWaT_Dataset_Normal_v0.csv',
                                                                    scaler_type='standar',
                                                                    save_npy=False)

X_train, X_val, y_train, y_val = train_test_split(
    train_data, labels, test_size=0.1, shuffle=False, random_state=42)

# --- model training --- #
model = MTSAD_CNN_VAE(sequence_length=60,
                      number_of_vars=44,
                      hidden_units_e=16,
                      hidden_units_d=16,
                      batch_size=32,
                      latent_dim=3,
                      epochs=200,
                      learning_rate=1e-3,
                      decay_rate=0.96,
                      decay_step=12000
                      )

model.train(X_train, X_val)

pd.DataFrame(model.vae.history.history).plot()
plt.grid()


# --- model testing --- #
test_data, labels_ts, scaler_ts, columns_names_ts, timestamp_ts, length_data_ts = samples.get_data(seq_length=60,
                                                                    shift_sample=60,
                                                                    aggregate_sec=1,
                                                                    drop=[' P202', 'P301', 'P401', 'P404', 'P502', 'P601', 'P603'],
                                                                    scaler=scaler, 
                                                                    path="data/test/SWaT_Dataset_Attack_v0.csv",
                                                                    start_time=None,
                                                                    save_npy=False,
                                                                    sep=';')

ix_anomaly, rec = model.detect(test_data, th=1e-3)

error = np.abs((test_data - rec))

score = error.reshape((error.shape[0]*error.shape[1], error.shape[2])).mean(axis=1)
score = np.nan_to_num(score, nan=100)
true = labels_ts.reshape((labels_ts.shape[0]*labels_ts.shape[1]))
true = np.nan_to_num(true, nan=1)

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

precision, recall, thresholds = precision_recall_curve(true, score)

fig = plt.figure()
plt.fill_between(recall, precision,  color="orange", alpha=0.2)
plt.plot(recall, precision, color="darkorange", alpha=0.6)

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall')
plt.grid()
