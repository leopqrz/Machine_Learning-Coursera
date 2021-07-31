#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:53:56 2020

@author: leonardoqueiroz
"""

import numpy as np


def mapFeature(X, order): # one or two  features
    out = []
    try:
        for i in range(order+1):
            for j in range(i+1):
                out = np.concatenate((out, (X[:,0]**(i-j))*(X[:,1]**j)))
        out = np.reshape(np.ravel(out), (-1,int(out.shape[0]/X[:,0].shape[0])), 'F')
    
    except Exception:
        for i in range(order+1):
            out = np.concatenate((out, (X[:,0]**i)))
        out = np.reshape(np.ravel(out), (-1,int(out.shape[0]/X[:,0].shape[0])), 'F')
        
    return out