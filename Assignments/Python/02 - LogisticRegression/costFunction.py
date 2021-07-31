#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 03:36:46 2020

@author: leonardoqueiroz
"""

import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y, lambda_reg = 0):
    m = len(y)
    grad = np.zeros(np.shape(theta))

    h = sigmoid(np.dot(X,theta))
    
    J = (-1/m) * sum(y*np.log(h) + (1-y)*np.log(1-h)) + (lambda_reg/(2*m)) * sum(theta[1:]**2) # np.dot(theta[1:].T , theta[1:])
    
    grad[0]  = (1/m) * np.dot(X[:,0].T, (h - y))
    grad[1:] = (1/m) * np.dot(X[:,1:].T, (h - y)) + (lambda_reg/m)*theta[1:]
    
    return J, grad