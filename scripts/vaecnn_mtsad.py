# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:26:42 2021

@author: gastong@fing.edu.uy
"""

# import comet_ml at the top of your file
from comet_ml import Experiment

#Create an experiment with your api key
experiment = Experiment(
    api_key="2mO83hWSVG9aCk8MSq1ANHK2A",
    project_name="multivariate-time-series-anomaly-detection",
    workspace="gastongarciagonzalez",
    auto_param_logging=True,
    auto_metric_logging=True
)

experiment.add_tags(['vae_cnn', 'swat'])

from zipfile import ZipFile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from anomalias.mtsad_vae import MTSAD_CNN_VAE 


#data
zip_file = ZipFile('../../data/SWaT.zip')
df_train = pd.read_csv(zip_file.open('SWaT/Physical/SWaT_Dataset_Normal_v1.csv'))
df_test = pd.read_csv(zip_file.open('SWaT/Physical/SWaT_Dataset_Attack_v0.csv'))
    #se quita las primeras 4 horas ya que presenta un transitorio.
df_train[' Timestamp'] = pd.to_datetime(df_train[' Timestamp'])
th_time = df_train[' Timestamp'][0] + pd.Timedelta('4H')
df_train = df_train[df_train[' Timestamp'] > th_time]


X = df_train.iloc[:,1:-1].values
y = df_train.iloc[:,-1].values
y[y=='Attack'] = 1
y[y=='Normal'] = 0
y = y.astype('int')
X_test = df_test.iloc[:,1:-1].values
df_test.iloc[:,-1].value_counts()
y_test = df_test.iloc[:,-1].values
y_test[y_test=='Attack'] = 1
y_test[y_test=='A ttack'] = 1
y_test[y_test=='Normal'] = 0
y_test = y_test.astype('int')

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, shuffle=False, random_state=42)

St = StandardScaler()
X_train = St.fit_transform(X_train)
X_val = St.transform(X_val)
X_test = St.transform(X_test)

experiment.log_dataset_info(name='SWaT_Normalv1_Attackv0')

window = 240
steps = window // 8
experiment.log_parameter('wondow', window)
experiment.log_parameter('seq_length', steps)

# Generated training sequences for use in the model.
def create_sequences(values, WINDOW=window, time_steps=steps):
    output = []
    for i in range(0, len(values) - window, time_steps):
        output.append(values[i : (i + WINDOW)])
    return np.stack(output)


X_train = create_sequences(X_train)
X_val = create_sequences(X_val)
X_test = create_sequences(X_test, time_steps=window)

# --- model training --- #
model = MTSAD_CNN_VAE(sequence_length=window,
                      number_of_vars=51,
                      hidden_units_e=32,
                      hidden_units_d=32,
                      batch_size=32,
                      latent_dim=18,
                      epochs=200,
                      learning_rate=1e-3,
                      decay_rate=0.96,
                      decay_step=12000
                      )

model.train(X_train, X_val)

pd.DataFrame(model.vae.history.history).plot()
plt.grid()

ix_anomaly, rec = model.detect(X_test, th=1e-3)

error = np.abs((X_test - rec))

score = error.reshape((error.shape[0]*error.shape[1], error.shape[2])).mean(axis=1)
# score = np.nan_to_num(score, nan=100)
# true = labels_ts.reshape((labels_ts.shape[0]*labels_ts.shape[1]))
# true = np.nan_to_num(true, nan=1)
y_test = y_test[:score.shape[0]]

precision, recall, thresholds = precision_recall_curve(y_test, score)
fig = plt.figure()
plt.fill_between(recall, precision,  color="orange", alpha=0.2)
plt.plot(recall, precision, color="darkorange", alpha=0.6)
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall (VAE_CNN)')
plt.grid()
experiment.log_figure('VAE_CNN_AD_precision_recall_curve', fig)

experiment.log_curve(f"recall_precision", recall, precision)
ave_precision = average_precision_score(y_test, score)
experiment.log_metric('AP', ave_precision)

f_score = 2*(precision * recall)/(precision + recall)

fig = plt.figure()
idx = thresholds < 2.5
plt.plot(thresholds[idx], f_score[:-1][idx])
plt.plot(thresholds[idx], precision[:-1][idx])
plt.plot(thresholds[idx], recall[:-1][idx])
plt.title('f1, precision, recall')
plt.legend(['F1', 'precision', 'recall'])
plt.xlabel('thresholds')
plt.grid()
experiment.log_figure('VAE_CNN_AD_f1_precision_recall', fig)

experiment.end()