# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:37:31 2021

@author: gastong@fing.edu.uy
"""

# import comet_ml at the top of your file
from comet_ml import Experiment
from AD import ISO_FOREST_AD, plot_score, precision_recall, predict
from datasets import swat, wadi

#data
data = 'swat'
if data=='swat':
    X_train, X_val, X_test, y_train, y_val, y_test = swat()
    labels=[1, 0]
elif data=='wadi':
    X_train, X_val, X_test, y_train, y_val, y_test = wadi()
    labels=[-1, 1]
    
#Create an experiment with your api key
experiment = Experiment(
    api_key="VZhK7C4klolOVuvJAQ1OrekYt",
    project_name="mts-anomaly-detection",
    workspace="gastong",
    auto_param_logging=False,
    auto_metric_logging=False,
)

experiment.set_name('isoforest')

n_est = 50
max_samp = 0.95
max_feat = 0.95
bootstrap = True

IF = ISO_FOREST_AD(n_est, max_samp, max_feat, bootstrap)
IF.fit(X_train)
#score_train = IF.score(X_train)
#score_val = IF.score(X_val)
score_test = IF.score(X_test)

ap, f1, precision, recall, f1_max_th, fig_pre_rec, fig_th_pre_rec = precision_recall(y_test, 
                                                       score_test,
                                                       limit=1000,
                                                       label_anomaly=labels[0])

#fig_score_train = plot_score(score_train, 'train')
#fig_score_val = plot_score(score_val, 'validation')
fig_score_test = plot_score(score_test, 'test', 10, labels=y_test, th=f1_max_th)

y_pred, conf_matrix = predict(score_test, f1_max_th, y_test, labels)

experiment.add_tags([data])
parameters = {'n_estimators': n_est, 'max_samples': max_samp,
              'max_features': max_feat, 'boostrap': bootstrap}
experiment.log_parameters(parameters)
experiment.log_metric('ap', ap)
experiment.log_metric('f1', f1)
experiment.log_metric('precision', precision)
experiment.log_metric('recall', recall)
experiment.log_metric('train_time', IF.time_)
experiment.log_parameter('th_f1', f1_max_th)
experiment.log_figure('score_test',fig_score_test)
experiment.log_figure('precision_recall',fig_pre_rec)
experiment.log_figure('th_pre_rec_f1', fig_th_pre_rec)
experiment.log_confusion_matrix(matrix=conf_matrix, labels=labels)

experiment.end()