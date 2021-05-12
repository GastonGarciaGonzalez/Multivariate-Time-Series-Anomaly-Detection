# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:43:16 2021

@author: gastong@fing.edu.uy
"""

from comet_ml import Experiment

#Create an experiment with your api key
experiment = Experiment(
    api_key="VZhK7C4klolOVuvJAQ1OrekYt",
    project_name="mts-anomaly-detection",
    workspace="gastong",
    auto_param_logging=False,
    auto_metric_logging=False,
)

experiment.set_name('mad_gan')

experiment.add_tags(['wadi'])
experiment.log_metric('f1', 0.37)
experiment.log_metric('precision', 0.4144)
experiment.log_metric('recall', 0.3392)
experiment.end()