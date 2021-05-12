# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:44:43 2021

@author: gastong@fing.edu.uy
"""
# import comet_ml at the top of your file
from comet_ml import Experiment
from AD import VAE_AD, plot_score, precision_recall
from datasets import swat, wadi


sequence_length=240
steps=60
fil1_conv_enc=8
fil2_conv_enc=16
unit_den_enc=16
fil1_conv_dec=16
fil2_conv_dec=8
kernel=3
strs=2
drop_rate=0.2
batch_size=32
latent_dim=9
epochs=100
learning_rate=1e-3
decay_rate=0.98
decay_step=12000
metric='mse'

#data
data = 'wadi'

if data=='swat':
    X_train, X_val, X_test, y_train, y_val, y_test = swat(validation_set='train',
                                                                  split=True,
                                                                  window=sequence_length,
                                                                  steps=steps)
    label=1
    number_of_vars=51

elif data=='wadi':
    X_train, X_val, X_test, y_train, y_val, y_test = wadi(validation_set='train',
                                                                  split=True,
                                                                  window=sequence_length,
                                                                  steps=steps)
    label=-1
    number_of_vars=123
    
#Create an experiment with your api key
experiment = Experiment(
    api_key="VZhK7C4klolOVuvJAQ1OrekYt",
    project_name="mts-anomaly-detection",
    workspace="gastong",
    auto_param_logging=False,
    auto_metric_logging=True,
)

experiment.set_name('vae')

vae = VAE_AD(sequence_length,
            number_of_vars,
            fil1_conv_enc,
            fil2_conv_enc,
            unit_den_enc,
            fil1_conv_dec,
            fil2_conv_dec,
            kernel,
            strs,
            drop_rate,
            batch_size,
            latent_dim,
            epochs,
            learning_rate,
            decay_rate,
            decay_step)

vae.fit(X_train, X_val)
#score_train = vae.score(X_train, metric)
#score_val = vae.score(X_val, metric)
score_test = vae.score(X_test, metric)

fig_loss = vae.plot_loss()

ap, f1_max, precision, recall, f1_max_th, fig_pre_rec, fig_th_pre_rec = precision_recall(y_test, 
                                                       score_test,
                                                       limit=1000,
                                                       label_anomaly=label)

#fig_score_train = plot_score(score_train, 'train', step=100)
#fig_score_val = plot_score(score_val, 'validation', step=100)
fig_score_test = plot_score(score_test, 'test', step=1, labels=y_test, th=f1_max_th)

experiment.add_tags([metric, data])
parameters = {'sequence_length':sequence_length,
                'steps':steps,
                'number_of_vars':number_of_vars,
                'fil1_conv_enc':fil1_conv_enc,
                'fil2_conv_enc':fil2_conv_enc,
                'unit_den_enc':unit_den_enc,
                'fil1_conv_dec':fil1_conv_dec,
                'fil2_conv_dec':fil2_conv_dec,
                'kernel':kernel,
                'strs':strs,
                'drop_rate':drop_rate,
                'batch_size':batch_size,
                'latent_dim':latent_dim,
                'epochs':epochs,
                'learning_rate':learning_rate,
                'decay_rate':decay_rate,
                'decay_step':decay_step,
                'metric':metric
}

experiment.log_parameters(parameters)
experiment.log_metric('ap', ap)
experiment.log_metric('f1', f1_max)
experiment.log_parameter('th_f1', f1_max_th)
experiment.log_metric('precision', precision)
experiment.log_metric('recall', recall)
experiment.log_metric('train_time', vae.time_)
experiment.log_parameter('th_f1', f1_max_th)
experiment.log_figure('losses',fig_loss)
experiment.log_figure('score_test',fig_score_test)
experiment.log_figure('precision_recall',fig_pre_rec)
experiment.log_figure('th_pre_rec_f1', fig_th_pre_rec)
experiment.end()

