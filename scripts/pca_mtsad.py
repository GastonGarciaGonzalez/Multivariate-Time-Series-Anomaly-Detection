# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:38:58 2021

@author: gastong@fing.edu.uy
"""
# import comet_ml at the top of your file
from comet_ml import Experiment
from AD import PCA_AD, plot_score, precision_recall, predict
from datasets import swat, wadi


#data
data = 'swat'
if data=='swat':
    X_train, X_val, X_test, y_train, y_val, y_test = swat(validation_set='test')
    labels=[1, 0]
elif data=='wadi':
    X_train, X_val, X_test, y_train, y_val, y_test = wadi(validation_set='test')
    labels=[-1, 1]
    
#Create an experiment with your api key
experiment = Experiment(
    api_key="VZhK7C4klolOVuvJAQ1OrekYt",
    project_name="mts-anomaly-detection",
    workspace="gastong",
    auto_param_logging=False,
    auto_metric_logging=False,
)

experiment.set_name('pca')

var = 0.8
metric='rmse'

pca = PCA_AD(var)
pca.fit(X_train)
#score_train = pca.score(X_train, metric)
#score_val = pca.score(X_val, metric)
score_test = pca.score(X_test, metric)



ap, f1_max, precision, recall, f1_max_th, fig_pre_rec, fig_th_pre_rec = precision_recall(y_test, 
                                                       score_test,
                                                       limit=1000,
                                                       label_anomaly=labels[0])

#fig_score_train = plot_score(score_train, 'train')
#fig_score_val = plot_score(score_val, 'validation')
fig_score_test = plot_score(score_test, 'test', 10, labels=y_test, th=f1_max_th)

fig_cumsum = pca.plot_cumsum()

y_pred, conf_matrix = predict(score_test, f1_max_th, y_test, labels)


experiment.add_tags([data, metric])
parameters = {'var': var, 'pc': pca.pcs_, 'metric': metric}
experiment.log_parameters(parameters)
experiment.log_metric('ap', ap)
experiment.log_metric('f1', f1_max)
experiment.log_metric('precision', precision)
experiment.log_metric('recall', recall)
experiment.log_metric('train_time', pca.time_)
experiment.log_parameter('th_f1', f1_max_th)
experiment.log_figure('cumsum', fig_cumsum)
experiment.log_figure('score_test',fig_score_test)
experiment.log_figure('precision_recall',fig_pre_rec)
experiment.log_figure('th_pre_rec_f1', fig_th_pre_rec)
experiment.log_confusion_matrix(matrix=conf_matrix, labels=labels)

experiment.end()
