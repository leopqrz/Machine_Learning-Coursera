#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:55:51 2020

@author: leonardoqueiroz
"""

import numpy as np


def costFunction(X, y, theta, lamb):
    m = len(y)
    h = np.dot(X,theta)

    # J = (1/(2*m)) * sum((h - y)**2 + lamb*theta**2)
    J = (1/(2*m)) * (np.dot((h-y).T, (h-y)) + lamb*sum(theta[1:]**2))
    
    return J
