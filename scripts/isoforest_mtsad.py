# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:37:31 2021

@author: gastong@fing.edu.uy
"""

# import comet_ml at the top of your file
from comet_ml import Experiment

#Create an experiment with your api key
experiment = Experiment(
    api_key="2mO83hWSVG9aCk8MSq1ANHK2A",
    project_name="multivariate-time-series-anomaly-detection",
    workspace="gastongarciagonzalez",
    auto_param_logging=False,
    auto_metric_logging=False,
)

experiment.add_tags(['isoforest', 'swat'])

from zipfile import ZipFile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


#data
zip_file = ZipFile('../data/SWaT.zip')
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

#%%
# Isolation Forest
from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=50, random_state=42, contamination=0.3).fit(X_train)
experiment.log_parameters({'n_estimators':50
                           , 'max_samples': 'auto'
                           , 'contamination': 0.3
                           , 'max_features': 1
                           , 'bootstrap': False})

score_train = -clf.score_samples(X_train)
fig = plt.figure(figsize=[18,3])
plt.scatter(range(len(score_train)), score_train, s=3)
plt.grid()
plt.title('score train')
plt.xlabel('index')
plt.ylabel('score_samples')
experiment.log_figure('IsoForest_score_train', fig)

score_val = -clf.score_samples(X_val)
fig = plt.figure(figsize=[18,3])
plt.scatter(range(len(score_val)), score_val, s=3, color="darkorange")
plt.grid()
plt.title('score validation')
plt.xlabel('index')
plt.ylabel('score_samples')
experiment.log_figure('IsoForest_score_validation', fig)


# h_X = pca.transform(X_test)
# X_rec = pca.inverse_transform(h_X)
# rmse_X_test = np.sqrt(mean_squared_error(X_test.T, X_rec.T, multioutput='raw_values'))
# plt.figure(figsize=[18,3])
# plt.scatter(range(len(rmse_X_test)), rmse_X_test, s=3, color="darkgreen")
# plt.grid()
# plt.legend(['Error test'])

score_test = -clf.score_samples(X_test)
fig = plt.figure(figsize=[18,3])
plt.scatter(range(len(score_test)), score_test, s=3, c=y_test, label=y_test)
plt.grid()
plt.title('score test')
plt.legend(['Normal_labels', 'Attack_labels'])
plt.xlabel('index')
plt.ylabel('rmse')
experiment.log_figure('IsoForest_score_test', fig)

precision, recall, thresholds = precision_recall_curve(y_test, score_test)
fig = plt.figure()
plt.fill_between(recall, precision,  color="blue", alpha=0.2)
plt.plot(recall, precision, color="darkblue", alpha=0.6)
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall (IsoForest)')
plt.grid()
experiment.log_figure('IsoForest_AD_precision_recall_curve', fig)

experiment.log_curve(f"recall_precision", recall, precision)
ave_precision = average_precision_score(y_test, score_test)
experiment.log_metric('AP', ave_precision)

f_score = 2*(precision * recall)/(precision + recall)

fig = plt.figure()
#idx = thresholds < 3.5
plt.plot(thresholds, f_score[:-1])
plt.plot(thresholds, precision[:-1])
plt.plot(thresholds, recall[:-1])
plt.title('f1, precision, recall')
plt.legend(['F1', 'precision', 'recall'])
plt.xlabel('thresholds')
plt.grid()
experiment.log_figure('IsoForest_AD_f1_precision_recall', fig)

experiment.end()