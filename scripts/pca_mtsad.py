# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:38:58 2021

@author: gastong@fing.edu.uy
"""
# import comet_ml at the top of your file
from comet_ml import Experiment
from AD import PCA_AD
from datasets import swat, wadi


#data
data = 'wadi'
if data=='swat':
    X_train, X_val, X_test, y_train, y_val, y_test = swat()
    label=1
elif data=='wadi':
    X_train, X_val, X_test, y_train, y_val, y_test = wadi()
    label=-1
    
#Create an experiment with your api key
experiment = Experiment(
    api_key="VZhK7C4klolOVuvJAQ1OrekYt",
    project_name="mts-anomaly-detection",
    workspace="gastong",
    auto_param_logging=False,
    auto_metric_logging=False,
)

var = 0.90
metric='rmse'

pca = PCA_AD(var)
pca.fit(X_train)
score_train = pca.score(X_train, metric)
score_val = pca.score(X_val, metric)
score_test = pca.score(X_test, metric)

fig_score_train = pca.plot_score(score_train, 'train')
fig_score_val = pca.plot_score(score_val, 'validation')
fig_score_test = pca.plot_score(score_test, 'test', labels=y_test)

ap, fig_pre_rec, fig_th_pre_rec = pca.precision_recall(y_test, 
                                                       score_test,
                                                       limit=1000,
                                                       label=label)

fig_cumsum = pca.plot_cumsum()

experiment.add_tags(['pca', data, metric])
parameters = {'var': var, 'pc': pca.pcs_, 'metric': metric}
experiment.log_parameters(parameters)
experiment.log_metric('ap', ap)
experiment.log_figure('cumsum', fig_cumsum)
experiment.log_figure('precision_recall',fig_pre_rec)
experiment.log_figure('th_pre_rec_f1', fig_th_pre_rec)
experiment.end()
