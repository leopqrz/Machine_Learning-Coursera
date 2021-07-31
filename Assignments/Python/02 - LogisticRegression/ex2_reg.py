#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:08:30 2020

@author: leonardoqueiroz
"""

''' Machine Learning Online Class - Exercise 2: Logistic Regression 
 
   Instructions
   ------------

   This file contains code that helps you get started on the second part
   of the exercise which covers regularization with logistic regression.
 
   You will need to complete the following functions in this exericse:
 
      sigmoid.m
      costFunction.m
      predict.m
      costFunctionReg.m
 
   For this exercise, you will not need to change any code in this file,
   or any other files other than those mentioned above.
'''

from matplotlib import pyplot as plt
import os
import numpy as np

from plotData import plotData
from mapFeature import mapFeature
from costFunction import costFunction
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict
''' ======================= Initialization ======================= '''

os.system('clear')  # For Linux/OS X

''' Load Data '''
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = np.loadtxt("ex2data2.txt", dtype='f', delimiter=',')

X = data[:,:2].reshape(-1,2)
y = data[:,2].reshape(-1,1)


plot = plotData(X,y)
# Labels and Legend
plot.xlabel('Microchip Test 1')
plot.ylabel('Microchip Test 2')
plot.legend(['y = 1', 'y = 0']) # Specified in plot order


plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

''' =========== Part 1: Regularized Logistic Regression ============ '''
os.system('clear')  # For Linux/OS X

#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).


# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled

X = mapFeature(X)        

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1],1)) # Start t0 and t1 (Parameters)  

# Set regularization parameter lambda to 1
lambda_reg = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunction(initial_theta, X, y, lambda_reg)

print('Cost at initial theta (zeros): %f\n'% cost)
print('Expected cost (approx): 0.693\n');
print('Gradient at initial theta (zeros) - first five values only:\n')
print(' %a \n' % np.round(grad[0:5], 4))
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1],1))
cost, grad = costFunction(test_theta, X,y, lambda_reg = 10)

print('\nCost at test theta (with lambda = 10): %f\n' % cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(' %a \n' % np.round(grad[0:5], 4))
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X


''' ============= Part 2: Regularization and Accuracies ============= '''
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1 (you should vary this)
lambda_reg = 1

y = y.flatten() 

def cost_reg(theta, X, y, lambda_reg):
    J,_ = costFunction(theta, X, y, lambda_reg)
    return J

def grad_reg(theta, X, y, lambda_reg):
    _,grad = costFunction(theta, X, y, lambda_reg)
    return grad.flatten()

from scipy.optimize import fmin_tnc

theta,_,_ = fmin_tnc(cost_reg, 
                x0 = initial_theta, 
                fprime = grad_reg, 
                args=(X, y, lambda_reg),
                disp = False)

cost = cost_reg(theta, X, y, lambda_reg)
theta = theta.reshape(-1,1)



''' Plot Boundary '''
plot = plotDecisionBoundary(theta, X, y)
plot.title('lambda = %f' % lambda_reg)

# Labels and Legend
plot.xlabel('Microchip Test 1')
plot.ylabel('Microchip Test 2')
plot.legend(['y = 1', 'y = 0', 'Decision boundary'])


plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X

# Compute accuracy on our training set 
p = predict(theta, X)
p = p.flatten()
print('Train Accuracy: %f\n' % (np.mean(np.double(p == y))*100))
print('Expected accuracy (approx): 83.1\n')
print('\n')









