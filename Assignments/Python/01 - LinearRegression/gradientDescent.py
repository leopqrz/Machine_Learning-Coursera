#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:56:34 2020

@author: leonardoqueiroz
"""

import numpy as np
from costFunction import costFunction


def gradientDescent(X, y, theta, alpha, num_iters, lamb): # alpha: Learning Rate, inter: Number of interactions
    ''' Setup the Parameters '''
    m = len(y)
    J_history = np.zeros(num_iters).reshape(-1,1)
    
    for i in range(num_iters): # Number of interactions
                
        h = np.dot(X,theta) # Hypothesis Function
        ''' Gradient Descent applied to the Cost Function '''
        # theta0 that doesn't need regularization
        
        t0 = theta[0][0] * (1 -  (alpha*0)/m) - (alpha/m) * np.dot(X.T, (h - y))[0][0]
        theta = theta * (1 -  (alpha*lamb)/m) - (alpha/m) * np.dot(X.T, (h - y))
        theta[0][0] = t0
        

        J_history[i] = costFunction(X, y, theta, lamb)

    return theta, J_history
