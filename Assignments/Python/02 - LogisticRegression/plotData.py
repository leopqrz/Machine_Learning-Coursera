#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 03:33:50 2020

@author: leonardoqueiroz
"""
import numpy as np
from matplotlib import pyplot as plt


def plotData(X,y):

    pos = np.where(y == 1)
    neg = np.where(y == 0)

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y')
    return plt