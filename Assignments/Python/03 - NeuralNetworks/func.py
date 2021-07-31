#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:01:17 2020

@author: leonardoqueiroz
"""
import numpy as np
from scipy.optimize import fmin_cg


def sigmoid(x):
    result = 1/(1 + np.exp(-x))
    return result


    


    






    


