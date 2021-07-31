#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:57:35 2020

@author: leonardoqueiroz
"""

import numpy as np

def normalEqn(X, y,lamb):
    n = X.shape[1]
    L = np.eye(n,n)
    L[0][0] = 0
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T , X) + lamb * L  ) , X.T) , y)
    # theta = pinv(X.T * X) * X.T * y
     
    return theta