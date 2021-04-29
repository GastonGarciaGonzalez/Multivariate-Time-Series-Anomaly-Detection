# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:57:21 2021

@author: usuario
"""
import numpy as np
import tensorflow as tf
import time
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.layers import Lambda, Input, Dense, Flatten, Reshape, Conv1D, Conv1DTranspose, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from anomalias import log

logger = log.logger( 'vaemodels' )


class MTSAD_VAE:
    
    def __init__(self, th, sequence_length=30, number_of_vars=1, hidden_units_e=100 
    ,hidden_units_d=100, batch_size=100, latent_dim=1, epochs=100, learning_rate=1e-2
    ,decay_rate=0.96, decay_step=1000):
        
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
        
        self.__th = th
        # network parameters
        self.__sequence_length = sequence_length
        self.__number_of_vars = number_of_vars
        self.__learning_rate = learning_rate
        self.__decay_rate = decay_rate
        self.__decay_step = decay_step
        self.__input_shape = (sequence_length, number_of_vars)
        self.__hidden_units_e = hidden_units_e
        self.__hidden_units_d = hidden_units_d
        self.__batch_size = batch_size
        self.__latent_dim = latent_dim
        self.__epochs = epochs    
        
        # VAE model = encoder + decoder
        
        # build encoder model
        self.__inputs = Input(shape=self.__input_shape, name='encoder_input')
        self.__inputs_flatten = Flatten()(self.__inputs)
        self.__h_enc = Dense(self.__hidden_units_e, activation='tanh')(self.__inputs_flatten)
        
        # reparameterization trick
        self.__z_mean = Dense(self.__latent_dim, name='z_mean')(self.__h_enc)
        self.__z_log_var = Dense(self.__latent_dim, name='z_log_var')(self.__h_enc)
        
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.__z = Lambda(sampling, output_shape=(self.__latent_dim,), name='z')([self.__z_mean, self.__z_log_var])
        
        # instantiate encoder model
        encoder = Model(self.__inputs, [self.__z_mean, self.__z_log_var, self.__z], name='encoder')
        logger.info('Encoder. \n %s', encoder.summary())
        
        
        # build decoder model
        self.__latent_inputs = Input(shape=(self.__latent_dim,), name='z_sampling')
        self.__h_z = Dense(self.__hidden_units_d, activation='tanh')(self.__latent_inputs)
        self.__x_recons_flatten = Dense(sequence_length * number_of_vars)(self.__h_z)
        self.__outputs = Reshape((sequence_length, number_of_vars))(self.__x_recons_flatten)
        
        # instantiate decoder model
        decoder = Model(self.__latent_inputs, self.__outputs, name='decoder')
        logger.info('Decoder. \n %s', decoder.summary())
        
        # instantiate VAE model
        self.__outputs = decoder(encoder(self.__inputs)[2])
        self.__vae = Model(self.__inputs, self.__outputs, name='vae_mlp')


    def train(self, train_data):

        reconstruction_loss = tf.reduce_mean(mse(self.__inputs, self.__outputs))
        reconstruction_loss *= (self.__sequence_length * self.__number_of_vars)
        kl_loss = 1 + self.__z_log_var - K.square(self.__z_mean) - K.exp(self.__z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.__vae.add_loss(vae_loss)
        
        lr_schedule = optimizers.schedules.ExponentialDecay(self.__learning_rate
                                                            , decay_steps=self.__decay_step
                                                            , decay_rate=self.__decay_rate
                                                            , staircase=True)
        
        opt = optimizers.Adam(learning_rate=lr_schedule)
        self.__vae.compile(optimizer=opt)
        logger.info('VAE_model. \n %s', self.__vae.summary())
        self.__vae.summary()
        
        time_start = time.time()
        self.__vae.fit(train_data
                , batch_size=self.__batch_size
                , epochs=self.__epochs)
        logger.info('Training time. \n %s', time.time() - time_start)
        
        
        
    def detect(self, observations, th):
        
        #obs_aux = np.expand_dims(observations, -1)
        reconstructions = self.__vae.predict(observations)
#        reconstructions = np.squeeze(reconstructions, axis=-1)
        
        
        error = (observations - reconstructions)**2
        idx_anom = error > th
        
        return idx_anom, reconstructions
    
    
    
    
class MTSAD_CNN_VAE:
    
    def __init__(self, sequence_length=30, number_of_vars=1, hidden_units_e=100 
    ,hidden_units_d=100, batch_size=100, latent_dim=1, epochs=100, learning_rate=1e-2
    ,decay_rate=0.96, decay_step=1000):
        
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
        self.sequence_length = sequence_length
        self.number_of_vars = number_of_vars
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.input_shape = (sequence_length, number_of_vars)
        self.hidden_units_e = hidden_units_e
        self.hidden_units_d = hidden_units_d
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epochs = epochs    
        
        # VAE model = encoder + decoder
        
        # build encoder model
        self.__inputs = Input(shape=self.input_shape, name='enc_input')
        self.__inputs_h_enc = Conv1D(16, 3, activation='tanh', strides=2, 
                                     padding="same", name='enc_conv1d_1')(self.__inputs)
        self.__inputs_h_enc = Dropout(0.2)(self.__inputs_h_enc)
        self.__inputs_h_enc = Conv1D(32, 3, activation='tanh', strides=2,
                                     padding="same", name='enc_conv1d_2')(self.__inputs_h_enc)
        self.__inputs_flatten = Flatten(name='enc_flatten')(self.__inputs_h_enc)
        self.__h_enc = Dense(self.hidden_units_e, activation='tanh',
                             name='enc_output')(self.__inputs_flatten)
        
        # reparameterization trick
        self.__z_mean = Dense(self.latent_dim, name='z_mean')(self.__h_enc)
        self.__z_log_var = Dense(self.latent_dim, name='z_log_var')(self.__h_enc)
        
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.__z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.__z_mean, self.__z_log_var])
        
        # instantiate encoder model
        encoder = Model(self.__inputs, [self.__z_mean, self.__z_log_var, self.__z], name='encoder')
        logger.info('Encoder. \n %s', encoder.summary())
        
        
        # build decoder model
        self.__latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        self.__h_z = Dense(60*32, activation='tanh', name='dec_input')(self.__latent_inputs)
        self.__h_z = Reshape((60, 32), name='dec_reshape')(self.__h_z)
        self.__h_z = Conv1DTranspose(32, 3, activation='tanh', strides=2, 
                                     padding="same", name='dec_conv1d_1')(self.__h_z)
        self.__inputs_h_enc = Dropout(0.2)(self.__h_z)
        self.__h_z = Conv1DTranspose(16, 3, activation='tanh', strides=2, 
                                     padding="same", name='dec_conv1d_2')(self.__h_z)
        self.__outputs = Conv1DTranspose(self.number_of_vars, 3, activation=None, 
                                         padding="same", name='dec_conv1d_output')(self.__h_z)
        #self.__x_recons_flatten = Dense(sequence_length * number_of_vars)(self.__h_z)
        #self.__outputs = Reshape((sequence_length, number_of_vars))(self.__x_recons_flatten)
        
        # instantiate decoder model
        decoder = Model(self.__latent_inputs, self.__outputs, name='decoder')
        logger.info('Decoder. \n %s', decoder.summary())
        
        # instantiate VAE model
        self.__outputs = decoder(encoder(self.__inputs)[2])
        self.vae = Model(self.__inputs, self.__outputs, name='vae_mlp')


    def train(self, train_data, val_data):

        reconstruction_loss = tf.reduce_mean(mse(self.__inputs, self.__outputs))
        reconstruction_loss *= (self.sequence_length * self.number_of_vars)
        kl_loss = 1 + self.__z_log_var - K.square(self.__z_mean) - K.exp(self.__z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        
        lr_schedule = optimizers.schedules.ExponentialDecay(self.learning_rate
                                                            , decay_steps=self.decay_step
                                                            , decay_rate=self.decay_rate
                                                            , staircase=True)
        
        opt = optimizers.Adam(learning_rate=lr_schedule)
        self.vae.compile(optimizer=opt)
        #logger.info('VAE_model. \n %s', self.__vae.summary())
        #self.__vae.summary()
        
        time_start = time.time()
        self.vae.fit(train_data
                , batch_size=self.batch_size
                , epochs=self.epochs
                , shuffle = True
                , validation_data = (val_data, val_data))
        logger.info('Training time. \n %s', time.time() - time_start)
        
        
        
    def detect(self, observations, th):
        
        #obs_aux = np.expand_dims(observations, -1)
        reconstructions = self.vae.predict(observations)
#        reconstructions = np.squeeze(reconstructions, axis=-1)
        
        
        error = (observations - reconstructions)**2
        idx_anom = error > th
        
        return idx_anom, reconstructions