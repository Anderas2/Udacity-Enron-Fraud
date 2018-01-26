# -*- coding: utf-8 -*-
"""
Little functions, extracted from the main text as i assume they may be
reusable sooner or later
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def print2d(X, Ax1, Ax2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,int(Ax1)], X[:,int(Ax2)])
    plt.show()

def hist(X):
    fig = plt.figure()
    for i,v in enumerate(range(X.shape[1])):
        v = v+1
        ax1 = fig.add_subplot(X.shape[1],1,v)
        ax1.hist(X[v], bins = 20)
    plt.show()

def heat(X, titles=[]):

    ax = plt.axes()
    ax.set_title('Correlations found')
    sns.heatmap(X,
                xticklabels = titles,
                yticklabels = titles)

def more_features(data_dict):
    data_pd = pd.DataFrame(data_dict)
    data_pd = data_pd.transpose()
    data_pd.fillna(value=0, inplace=True)
    data_pd.replace(to_replace='NaN', value=0, inplace=True)
    original_pd = data_pd.copy()
    data_pd.drop('email_address',axis=1,inplace=True)
    data_pd.drop('poi',axis=1,inplace=True)
    for column in data_pd.columns:
        for column2 in data_pd.columns:
            original_pd[column + 'X' + column2] = data_pd.apply(
                    lambda row:(row[column]*row[column2]), axis = 1)
            original_pd[column + '-' + column2] = data_pd.apply(
                    lambda row:(row[column]-row[column2]), axis = 1)
            original_pd[column + '/' + column2] = data_pd.apply(
                    lambda row:(row[column]/row[column2]), axis = 1)
    original_pd.fillna(value=0, inplace=True)
    original_pd.replace(to_replace=np.inf, value=0, inplace=True)
    original_pd.replace(to_replace=-np.inf, value=0, inplace=True)
    return original_pd
