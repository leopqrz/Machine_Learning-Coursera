#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 03:28:45 2020

@author: leonardoqueiroz
"""

''' Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all 

     Instructions
     ------------

     This file contains code that helps you get started on the
     linear exercise. You will need to complete the following functions
     in this exercise:

        lrCostFunction.py (logistic regression cost function)
        oneVsAll.py
        predictOneVsAll.py
        predict.py

     For this exercise, you will not need to change any code in this file,
     or any other files other than those mentioned above.
'''


import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import os

from displayData import displayData
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll



''' ======================= Initialization ======================= '''

''' Setup the parameters you will use for this part of the exercise '''
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                         # (note that we have mapped "0" to label 10)



''' =========== Part 1: Loading and Visualizing Data ============= '''
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

os.system('clear')  # clear: For Linux/OS X / clc: For Windows

# Load Training Data
print('Loading and Visualizing Data ...\n')
data = loadmat('ex3data1')
X, y = data['X'], data['y']
# The matrices X and y will now be in your Python environment

m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]

displayData(sel)

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')



''' ============ Part 2a: Vectorize Logistic Regression ============ '''
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.

os.system('clear')  # clear: For Linux/OS X / clc: For Windows

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2]).reshape(-1,1)
X_t = np.append(np.ones((5,1)), np.linspace(0.1,1.5,15).reshape(5,3,order='F'), axis=1)
y_t = np.array([1,0,1,0,1]).reshape(-1,1)
lambda_t = 3
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('\nCost: %f\n' % J)
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(' %a \n' % grad)
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')



''' ============ Part 2b: One-vs-All Training ============ '''
os.system('clear')  # For Linux/OS X

print('\nTraining One-vs-All Logistic Regression...\n')

lambda_reg = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_reg)

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')



''' ================ Part 3: Predict for One-Vs-All ================ '''
os.system('clear')  # For Linux/OS X
pred = predictOneVsAll(all_theta, X)
print('\nTraining Set Accuracy: %f\n' % (np.mean(np.double(pred == y))*100))

