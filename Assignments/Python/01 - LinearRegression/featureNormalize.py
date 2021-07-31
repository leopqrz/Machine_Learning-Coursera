#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:55:08 2020

@author: leonardoqueiroz
"""

import numpy as np


def featureNormalize(X):
    mu = np.mean(X)
    std = np.std(X)
    X = X-mu/std
    
    return X, mu, std
