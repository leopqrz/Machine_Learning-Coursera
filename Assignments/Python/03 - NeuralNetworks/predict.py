#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:41:19 2020

@author: leonardoqueiroz
"""

import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    
    m = X.shape[0]
    p = np.zeros((m, 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m,1)), X))

    # Calculate the Hidden Layer
    a = sigmoid(np.dot(X,Theta1.T)) # hidden Layer
    
    # Add ones to the a data matrix
    a = np.column_stack((np.ones((m,1)), a))

    p = sigmoid(np.dot(a,Theta2.T)) # output Layer
    p = np.argmax(p,1)+1
    p = p.reshape(-1,1)

    return p

