# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 21:26:50 2021

@author: gastong@fing.edu.uy
"""


from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def create_sequences(values, WINDOW, time_steps):
    output = []
    for i in range(0, len(values) - WINDOW, time_steps):
        output.append(values[i : (i + WINDOW)])
    return np.stack(output)


def swat(zip_path='../../data/SWaT.zip',
        train_file='SWaT/Physical/SWaT_Dataset_Normal_v1.csv',
        test_file='SWaT/Physical/SWaT_Dataset_Attack_v0.csv',
        split=False,
        window=None,
        steps=None,
        ):
    
    zip_file = ZipFile(zip_path)
    df_train = pd.read_csv(zip_file.open(train_file))
    df_test = pd.read_csv(zip_file.open(test_file))
    
    #The transitional regime is dropped
    df_train[' Timestamp'] = pd.to_datetime(df_train[' Timestamp'])
    th_time = df_train[' Timestamp'][0] + pd.Timedelta('4H')
    df_train = df_train[df_train[' Timestamp'] > th_time]
    
    #values 
    X = df_train.iloc[:,1:-1].values
    y = df_train.iloc[:,-1].values
    y[y=='Attack'] = 1
    y[y=='Normal'] = 0
    y = y.astype('int')
    
    X_test = df_test.iloc[:,1:-1].values
    y_test = df_test.iloc[:,-1].values
    y_test[y_test=='Attack'] = 1
    y_test[y_test=='A ttack'] = 1
    y_test[y_test=='Normal'] = 0
    y_test = y_test.astype('int')
    
    #train validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, shuffle=False, random_state=42)
    
    #scaler
    St = StandardScaler()
    X_train = St.fit_transform(X_train)
    X_val = St.transform(X_val)
    X_test = St.transform(X_test)
    
    #windows
    if split:
        X_train = create_sequences(X_train, window, steps)
        X_val = create_sequences(X_val, window, steps)
        X_test = create_sequences(X_test, window, time_steps=window)
        
    return X_train, X_val, X_test, y_train, y_val, y_test


def wadi(zip_path='../../data/WADI_A2_19_Nov_2019.zip',
        train_file='WADI_A2_19_Nov_2019/WADI_14days_new.csv',
        test_file='WADI_A2_19_Nov_2019/WADI_attackdataLABLE.csv',
        split=False,
        window=None,
        steps=None,
        si_strategy='mean'
        ):
    
    zip_file = ZipFile(zip_path)
    df_train = pd.read_csv(zip_file.open(train_file))
    df_test = pd.read_csv(zip_file.open(test_file), header=1)
    
    #The transitional regime is dropped
    # df_train[' Timestamp'] = pd.to_datetime(df_train['Date'] + ' ' + df_train['Time'])
    # th_time = df_train[' Timestamp'][0] + pd.Timedelta('4H')
    # df_train = df_train[df_train[' Timestamp'] > th_time]
    
    #values 
    X = df_train.iloc[:,3:].values
    y = np.ones(X.shape[0])
    
    X_test = df_test.iloc[:-2,3:-1].values
    y_test = df_test.iloc[:-2,-1].values
    
    #train validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, shuffle=False, random_state=42)
    
    imp_mean = SimpleImputer(missing_values=np.nan, strategy=si_strategy)
    X_train = imp_mean.fit_transform(X_train)
    X_val = imp_mean.transform(X_val)
    X_test = imp_mean.transform(X_test)
    
    #scaler
    St = StandardScaler()
    X_train = St.fit_transform(X_train)
    X_val = St.transform(X_val)
    X_test = St.transform(X_test)
    
    #windows
    if split:
        X_train = create_sequences(X_train, window, steps)
        X_val = create_sequences(X_val, window, steps)
        X_test = create_sequences(X_test, window, time_steps=window)
        
    return X_train, X_val, X_test, y_train, y_val, y_test
        