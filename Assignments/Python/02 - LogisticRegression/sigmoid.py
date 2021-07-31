#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 03:35:03 2020

@author: leonardoqueiroz
"""

import numpy as np


def sigmoid(x):
    result = 1/(1 + np.exp(-x))
    return result
