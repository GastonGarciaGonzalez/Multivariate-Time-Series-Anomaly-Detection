# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:41:21 2021

@author: gastong@fing.edu.uy
"""
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, mean_squared_log_error, mean_poisson_deviance, mean_gamma_deviance
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import time
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.layers import Lambda, Input, Dense, Flatten, Reshape, Conv1D, Conv1DTranspose, Dropout, SimpleRNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers


def score_rec(metric, X, X_):
    
    if metric=='rmse':
        score = np.sqrt(mean_squared_error(X.T, X_.T, multioutput='raw_values'))
    elif metric=='mse':
        score = mean_squared_error(X.T, X_.T, multioutput='raw_values')
    elif metric=='mae':
        score = mean_absolute_error(X.T, X_.T, multioutput='raw_values')
    elif metric=='msle':
        score = mean_squared_log_error(X.T, X_.T, multioutput='raw_values')
    elif metric=='evs':
        score = explained_variance_score(X.T, X_.T, multioutput='raw_values')
    elif metric=='poisson':
        n = X.shape[0]
        score = np.zeros(n)
        X = np.abs(X)
        X_ = np.abs(X_)
        for i in range(n):
            score[i] = mean_poisson_deviance(X[i,:], X_[i,:])
    elif metric=='gamma':
        n = X.shape[0]
        score = np.zeros(n)
        X = np.abs(X)
        X_ = np.abs(X_)
        for i in range(n):
            score[i] = mean_gamma_deviance(X[i,:], X_[i,:])
        
    return score
    
def plot_score(score, data, step=1, labels=np.array([]), th=None):
    score = score[::step]
    if labels.shape[0]==0:
        labels=None
    else:
        labels = labels[::step]
        labels = labels[:score.shape[0]]
    fig_score = plt.figure(figsize=[18,3])
    plt.scatter(range(len(score)), score, s=3, c=labels)
    if th != None:
        plt.axhline(th, linewidth=1.0, c='red')
    plt.grid()
    plt.title('Score ' + data)
    plt.xlabel('index')
    plt.ylabel('score')
    
    return fig_score

def precision_recall(y, score, limit=1000, label_anomaly=1):
    y = y[:score.shape[0]]
    precision, recall, thresholds = precision_recall_curve(y, score, pos_label=label_anomaly)
    fig_pre_rec = plt.figure()
    plt.fill_between(recall, precision,  color="green", alpha=0.2)
    plt.plot(recall, precision, color="darkgreen", alpha=0.6)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Precision-Recall')
    plt.grid()
    
    ave_precision = average_precision_score(y, score, pos_label=label_anomaly)
    f_score = 2*(precision * recall)/(precision + recall)
    f_score = np.nan_to_num(f_score)
    f_score_max = np.max(f_score)
    ix_th = np.argmax(f_score)
    th = thresholds[ix_th]
    pre_f1max = precision[ix_th]
    rec_f1max = recall[ix_th] 
    
    fig_th_pre_rec = plt.figure()
    idx = thresholds < limit
    plt.plot(thresholds[idx], f_score[:-1][idx])
    plt.plot(thresholds[idx], precision[:-1][idx])
    plt.plot(thresholds[idx], recall[:-1][idx])
    plt.axvline(th, linewidth=1.0, c="r")
    plt.title('f1, precision, recall')
    plt.legend(['F1', 'precision', 'recall'])
    plt.xlabel('thresholds')
    plt.grid()
    
    return ave_precision, f_score_max, pre_f1max, rec_f1max, th, fig_pre_rec, fig_th_pre_rec

def predict(score, th, y_true, labels):
    output = np.where(score > th, labels[0], labels[1])
    matrix = confusion_matrix(y_true, output, labels)
    
    return output, matrix
    


class PCA_AD:
    
    def __init__(self, per_variance=None):
        self.__pca = PCA(n_components=per_variance)
        
    def fit(self, X):
        t_start = time.time()
        self.__pca.fit(X)
        self.time_ = (time.time()-t_start)/60
        self.cumsum_ = np.cumsum(self.__pca.explained_variance_ratio_)
        self.pcs_ = self.__pca.n_components_
        
        return self
    
    def score(self, X, metric='rmse'):
        rec = self.__pca.inverse_transform(self.__pca.transform(X))
        score = score_rec(metric, X, rec)
        
        return score
    
    def plot_cumsum(self):
        fig = plt.figure()
        plt.plot(self.cumsum_)
        plt.xlabel('d')
        plt.ylabel('variance ratio')
        plt.title('variance PCA')
        plt.grid()
    
        return fig
    
               
            
    
class ISO_FOREST_AD:
    
    def __init__(self, n_estimators=100, max_samples='auto', max_features=1.0, bootstrap=False, n_jobs=-1):
        self.__isoforest = IsolationForest(n_estimators=n_estimators,
                                           max_samples=max_samples,
                                           max_features=max_features,
                                           bootstrap=bootstrap,
                                           n_jobs=n_jobs)
        
    def fit(self, X):
        t_start = time.time()
        self.__isoforest.fit(X)
        self.time_ = (time.time()-t_start)/60
        
        return self
    
    def score(self, X):
        # score: The higher, the more abnormal.
        score = -self.__isoforest.score_samples(X)
        
        return score
    
    
    

class VAE_AD:
    
    def __init__(self,
                 sequence_length=30,
                 number_of_vars=10,
                 fil1_conv_enc=16,
                 fil2_conv_enc=32,
                 unit_den_enc=16,
                 fil1_conv_dec=32,
                 fil2_conv_dec=16,
                 kernel=3,
                 strs=2,
                 drop_rate=0.2,
                 batch_size=100,
                 latent_dim=1,
                 epochs=100,
                 learning_rate=1e-2,
                 decay_rate=0.96,
                 decay_step=1000,
                 ):
        
        # reparameterization trick
        # instead of sampling from Q(z|X), sample epsilon = N(0,I)
        # z = z_mean + sqrt(var) * epsilon
        def sampling(args):
            """Reparameterization trick by sampling from an isotropic unit Gaussian.
        
            # Arguments
                args (tensor): mean and log of variance of Q(z|X)
        
            # Returns
                z (tensor): sampled latent vector
            """
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean = 0 and std = 1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        # network parameters
        input_shape = (sequence_length, number_of_vars)
        self.batch_size = batch_size
        self.epochs = epochs    
        
        # VAE model = encoder + decoder
        
        # build encoder model
        inputs = Input(shape=input_shape, name='enc_input')
        h_enc = Conv1D(fil1_conv_enc, kernel, activation='tanh', strides=strs, 
                       padding="same", name='enc_conv1d_1')(inputs)
        h_enc = Dropout(drop_rate)(h_enc)
        h_enc = Conv1D(fil2_conv_enc, kernel, activation='tanh', strides=strs, 
                       padding="same", name='enc_conv1d_2')(h_enc)
        h_enc = Flatten(name='enc_flatten')(h_enc)
        h_enc = Dense(unit_den_enc, activation='tanh', 
                      name='enc_output')(h_enc)
        
        # reparameterization trick
        z_mean = Dense(latent_dim, name='z_mean')(h_enc)
        z_log_var = Dense(latent_dim, name='z_log_var')(h_enc)
        
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        
        
        # build decoder model
        reduce = sequence_length//(strs*2)
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        h_dec = Dense(reduce*fil1_conv_dec, activation='tanh', name='dec_input')(latent_inputs)
        h_dec = Reshape((reduce, fil1_conv_dec), name='dec_reshape')(h_dec)
        h_dec = Conv1DTranspose(fil1_conv_dec, kernel, activation='tanh', strides=strs, 
                                padding="same", name='dec_conv1d_1')(h_dec)
        h_dec = Dropout(drop_rate)(h_dec)
        h_dec = Conv1DTranspose(fil2_conv_dec, kernel, activation='tanh', strides=strs, 
                                padding="same", name='dec_conv1d_2')(h_dec)
        outputs = Conv1DTranspose(number_of_vars, kernel, activation=None, 
                                  padding="same", name='dec_conv1d_output')(h_dec)
       
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        
        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')
        
        reconstruction_loss = tf.reduce_mean(mse(inputs, outputs))
        reconstruction_loss *= (sequence_length * number_of_vars)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        
        lr_schedule = optimizers.schedules.ExponentialDecay(learning_rate,
                                                            decay_steps=decay_step,
                                                            decay_rate=decay_rate,
                                                            staircase=True,
                                                            )
        
        opt = optimizers.Adam(learning_rate=lr_schedule)
        self.vae.compile(optimizer=opt)

    def fit(self, X_train, X_val):
        t_start = time.time()
        self.history_ = self.vae.fit(X_train,
                     batch_size=self.batch_size,
                     epochs=self.epochs,
                     shuffle = True,
                     validation_data = (X_val, X_val),
                     )  
        self.time_ = (time.time() - t_start)/60
        return self
                
    def score(self, X, metric='rmse'):
        rec = self.vae.predict(X)
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
        rec = rec.reshape((rec.shape[0]*rec.shape[1], rec.shape[2]))
        score = score_rec(metric, X, rec)
        
        return score
    
    def plot_loss(self):
        fig = plt.figure()
        pd.DataFrame(self.history_.history).plot()
        plt.grid()
        
        return fig
    
    
    
    
class RNN_AD:
    
    def __init__(self,
                 sequence_length=30,
                 number_of_vars=10,
                 unit1=20,
                 unit2=20,
                 drop_rate=0.2,
                 batch_size=100,
                 epochs=100,
                 learning_rate=1e-3,
                 decay_rate=0.98,
                 decay_step=1000,
                 ):
        
        # network parameters
        self.batch_size = batch_size
        self.epochs = epochs  
            
        self.rnn = Sequential([
            SimpleRNN(unit1, return_sequences=True, input_shape=[sequence_length, number_of_vars]),
            SimpleRNN(unit2),
            Dense(number_of_vars)
            ])
        # input_shape = (sequence_length, number_of_vars)
        # self.rnn = Sequential()
        # self.rnn.add(Input(shape=input_shape, name='input'))
        # self.rnn.add(SimpleRNN(unit1, return_sequences=True, name='rnn1'))
        # self.rnn.add(SimpleRNN(unit2, name='rnn2'))
        # self.rnn.add(Dense(number_of_vars, name='output'))
    
        lr_schedule = optimizers.schedules.ExponentialDecay(learning_rate,
                                                       decay_steps=decay_step,
                                                       decay_rate=decay_rate,
                                                       staircase=True,
                                                       )
        
        opt = optimizers.Adam(learning_rate=lr_schedule)
        self.rnn.compile(loss='mse', optimizer=opt)
        self.rnn.summary()
    
    def fit(self, X_train, y_train, X_val, y_val):
        t_start = time.time()
        self.history_ = self.rnn.fit(X_train,
                                y_train,
                                epochs=self.epochs,
                                validation_data = (X_val, y_val),
                                )
        self.time_ = time.time() - t_start
        return self
     
    def score(self, X, y, metric='rmse'):
        rec = self.rnn.predict(X)
        score = score_rec(metric, y, rec)
        
        return score
    
    def plot_loss(self):
        fig = plt.figure()
        pd.DataFrame(self.history_.history).plot()
        plt.grid()
        
        return fig