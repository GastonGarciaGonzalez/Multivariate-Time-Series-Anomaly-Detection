# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:23:01 2021

@author: gastong@fing.edu.uy
"""

# import comet_ml at the top of your file
from comet_ml import Experiment
from AD import RNN_AD, plot_score, precision_recall, predict
from datasets import swat, wadi
from sklearn.metrics import mean_squared_error
import numpy as np

sequence_length=240
steps=10
unit1=20
unit2=20
drop_rate=0.2
batch_size=32
epochs=50
learning_rate=1e-3
decay_rate=0.98
decay_step=12000
metric='mse'

#data
data = 'swat'

if data=='swat':
    X_train, X_val, X_test, y_train, y_val, y_test = swat(validation_set='train',
                                                                  split=True,
                                                                  window=sequence_length,
                                                                  steps=steps,
                                                                  rnn=True)
    labels=[1, 0]
    number_of_vars=51
elif data=='wadi':
    X_train, X_val, X_test, y_train, y_val, y_test = wadi(validation_set='train',
                                                                  split=True,
                                                                  window=sequence_length,
                                                                  steps=steps,
                                                                  rnn=True)
    labels=[-1, 1]
    number_of_vars=123
    
#Create an experiment with your api key
experiment = Experiment(
    api_key="VZhK7C4klolOVuvJAQ1OrekYt",
    project_name="mts-anomaly-detection",
    workspace="gastong",
    auto_param_logging=False,
    auto_metric_logging=True,
)

experiment.set_name('rnn')

rnn = RNN_AD(sequence_length,
            number_of_vars,
            unit1,
            unit2,
            drop_rate,
            batch_size,
            epochs,
            learning_rate,
            decay_rate,
            decay_step)

rnn.fit(X_train, y_train, X_val, y_val)
#score_train = rnn.score(X_train, y_train)
#score_val = rnn.score(X_val, y_val)
fig_loss = rnn.plot_loss()

#SCORE TEST
score_test = np.array([])
batch_aux = 0
n_test = y_test.shape[0]
while  score_test.shape[0] != y_test[sequence_length:].shape[0]:
    X_batch = np.empty((0, sequence_length, number_of_vars))
    y_batch = np.empty((0, number_of_vars))
    for i in range(batch_aux, min(batch_aux + batch_size, n_test-sequence_length)):
        X_aux = X_test[i : (i + sequence_length)]
        X_aux = X_aux.reshape((1, X_aux.shape[0], X_aux.shape[1]))
        y_aux = X_test[i + sequence_length + 1]
        y_aux = y_aux.reshape((1, y_aux.shape[0]))
        X_batch = np.concatenate((X_batch, X_aux))
        y_batch = np.concatenate((y_batch, y_aux))
        #print(i)
        
    y_pred = rnn.rnn.predict(X_batch)
    score_aux = np.sqrt(mean_squared_error(y_batch.T, y_pred.T, multioutput='raw_values'))
    score_test = np.concatenate((score_test, score_aux))
    batch_aux += batch_size
    
y_test = y_test[sequence_length:]

ap, f1, precision, recall, f1_max_th, fig_pre_rec, fig_th_pre_rec = precision_recall(y_test, 
                                                       score_test,
                                                       limit=1000,
                                                       label_anomaly=labels[0])

#fig_score_train = plot_score(score_train, 'train')
#fig_score_val = plot_score(score_val, 'validation')
fig_score_test = plot_score(score_test, 'test', 10, labels=y_test, th=f1_max_th)

y_pred, conf_matrix = predict(score_test, f1_max_th, y_test, labels)

experiment.add_tags([data, 'rmse'])
parameters = {'sequence_length':sequence_length,
                 'number_of_vars':number_of_vars,
                 'unit1':unit1,
                 'unit2':unit2,
                 'drop_rate':drop_rate,
                 'batch_size':batch_size,
                 'epochs':epochs,
                 'learning_rate':learning_rate,
                 'decay_rate':decay_rate,
                 'decay_step':decay_step}
experiment.log_parameters(parameters)
experiment.log_metric('ap', ap)
experiment.log_metric('f1', f1)
experiment.log_metric('precision', precision)
experiment.log_metric('recall', recall)
experiment.log_metric('train_time', rnn.time_)
experiment.log_parameter('th_f1', f1_max_th)
experiment.log_figure('losses',fig_loss)
experiment.log_figure('score_test',fig_score_test)
experiment.log_figure('precision_recall',fig_pre_rec)
experiment.log_figure('th_pre_rec_f1', fig_th_pre_rec)
experiment.log_confusion_matrix(matrix=conf_matrix, labels=labels)
experiment.end()
