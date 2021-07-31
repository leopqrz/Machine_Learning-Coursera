#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 03:35:38 2020

@author: leonardoqueiroz
"""

import numpy as np


def mapFeature(X, degree = 6):
    out = []
    for i in range(degree+1):
        for j in range(i+1):
            out = np.concatenate((out, (X[:,0]**(i-j))*(X[:,1]**j)))
    out = np.reshape(np.ravel(out), (-1,int(out.shape[0]/X[:,0].shape[0])), 'F')
    return out
