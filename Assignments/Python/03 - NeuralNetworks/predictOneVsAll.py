#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 03:05:09 2020

@author: leonardoqueiroz
"""
import numpy as np
from sigmoid import sigmoid

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    p = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m,1)), X))
    
    p = sigmoid(np.dot(X, all_theta.T))
    p = np.argmax(p,1)+1
    p = p.reshape(-1,1)
    
    return p