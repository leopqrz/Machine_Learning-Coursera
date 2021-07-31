#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 03:39:13 2020

@author: leonardoqueiroz
"""
import numpy as np
from sigmoid import sigmoid

def predict(theta, X):
    
    p = np.round(sigmoid(np.dot(X,theta)))
    
    return p