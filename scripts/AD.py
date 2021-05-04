# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:41:21 2021

@author: gastong@fing.edu.uy
"""

from sklearn.decomposition import PCA
import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, mean_squared_log_error
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

class PCA_AD:
    
    def __init__(self, per_variance=None):
        self.__pca = PCA(n_components=per_variance)
        
    def fit(self, X):
        self.__pca.fit(X)
        self.cumsum_ = np.cumsum(self.__pca.explained_variance_ratio_)
        self.pcs_ = self.__pca.n_components_
        
        return self
    
    def score(self, X, metric='rmse'):
        rec = self.__pca.inverse_transform(self.__pca.transform(X))
        
        if metric=='rmse':
            score = np.sqrt(mean_squared_error(X.T, rec.T, multioutput='raw_values'))
        elif metric=='mse':
            score = mean_squared_error(X.T, rec.T, multioutput='raw_values')
        elif metric=='mae':
            score = mean_absolute_error(X.T, rec.T, multioutput='raw_values')
        elif metric=='msle':
            score = mean_squared_log_error(X.T, rec.T, multioutput='raw_values')
        elif metric=='evs':
            score = explained_variance_score(X.T, rec.T, multioutput='raw_values')
            
        return score
    
    def plot_score(self, score, data, labels=None):
        fig_score = plt.figure(figsize=[18,3])
        plt.scatter(range(len(score)), score, s=3, c=labels)
        plt.grid()
        plt.title('Score ' + data)
        plt.xlabel('index')
        plt.ylabel('score')
        
        return fig_score
    
    def precision_recall(self, y, score, limit=0.35, label=1):
        precision, recall, thresholds = precision_recall_curve(y, score, pos_label=label)
        fig_pre_rec = plt.figure()
        plt.fill_between(recall, precision,  color="green", alpha=0.2)
        plt.plot(recall, precision, color="darkgreen", alpha=0.6)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.title('Precision-Recall')
        plt.grid()
        
        ave_precision = average_precision_score(y, score, pos_label=label)
        f_score = 2*(precision * recall)/(precision + recall)
        
        fig_th_pre_rec = plt.figure()
        idx = thresholds < limit
        plt.plot(thresholds[idx], f_score[:-1][idx])
        plt.plot(thresholds[idx], precision[:-1][idx])
        plt.plot(thresholds[idx], recall[:-1][idx])
        plt.title('f1, precision, recall')
        plt.legend(['F1', 'precision', 'recall'])
        plt.xlabel('thresholds')
        plt.grid()
        
        return ave_precision, fig_pre_rec, fig_th_pre_rec
    
    def plot_cumsum(self):
        fig = plt.figure()
        plt.plot(self.cumsum_)
        plt.xlabel('d')
        plt.ylabel('variance ratio')
        plt.title('variance PCA')
        plt.grid()
    
        return fig

    
