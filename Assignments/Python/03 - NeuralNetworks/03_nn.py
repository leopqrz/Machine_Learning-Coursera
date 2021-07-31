#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:29:11 2020

@author: leonardoqueiroz
"""

''' Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

   Instructions
   ------------
 
   This file contains code that helps you get started on the
   linear exercise. You will need to complete the following functions 
   in this exericse:

      lrCostFunction.m (logistic regression cost function)
      oneVsAll.m
      predictOneVsAll.m
      predict.m

   For this exercise, you will not need to change any code in this file,
   or any other files other than those mentioned above.
'''


import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import os

from displayData import displayData
from predict import predict

''' ======================= Initialization ======================= '''

''' Setup the parameters you will use for this exercise '''
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
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

''' ================ Part 2: Loading Pameters ================ '''
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
# Load saved matrices from file
weights = loadmat('ex3weights')
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# The matrices Theta1 and Theta2 will now be in your Octave
# environment
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26


plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')


''' ================= Part 3: Implement Predict ================= '''
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

os.system('clear')  # clear: For Linux/OS X / clc: For Windows

pred = predict(Theta1, Theta2, X)
print('\nTraining Set Accuracy: %f\n' % (np.mean(np.double(pred == y)) * 100))


plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')


#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(m)

for i in range(1,m+1):
    # Display 
    print('\nDisplaying Example Image\n')
    displayData(X[rp[i], :])

    pred = predict(Theta1, Theta2, X[rp[i],:].reshape(1,-1))
    print('\nNeural Network Prediction: %d (digit %d)\n' % (pred, pred%10))
    
    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
      break