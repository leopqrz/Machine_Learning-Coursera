#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:08:30 2020

@author: leonardoqueiroz
"""

''' Machine Learning Online Class - Exercise 2: Logistic Regression

    Instructions
    ------------
 
    This file contains code that helps you get started on the logistic
    regression exercise. You will need to complete the following functions 
    in this exericse:
  
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
from costFunction import costFunction
from plotDecisionBoundary import plotDecisionBoundary
from sigmoid import sigmoid
from predict import predict

''' ======================= Initialization ======================= '''

''' Load Data '''
# The first two columns contains the exam scores and the third column
# contains the label.

data = np.loadtxt("ex2data1.txt", dtype='f', delimiter=',')

X = data[:,:2].reshape(-1,2)
y = data[:,2].reshape(-1,1)
m = len(y)



''' ==================== Part 1: Plotting ==================== '''
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

os.system('clear')  # For Linux/OS X

print('Plotting data with + indicating (y = 1) examples and o \
      indicating (y = 0) examples.\n')


plot = plotData(X,y)
# Put some labels 
plot.xlabel('Exam 1 score')
plot.ylabel('Exam 2 score')
plot.legend(['Not Admitted', 'Admitted']) # Specified in plot order


plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X


''' ============ Part 2: Compute Cost and Gradient ============ '''
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = np.shape(X)

# Add intercept term to X
X = np.column_stack((np.ones((m,1)), X))

# Initialize fitting parameters
initial_theta = np.zeros((n+1, 1)) # Start t0 and t1 (Parameters)   

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): %f\n' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(' %a \n' % grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X

# Compute and display Cost and Gradient with non-zero theta 
test_theta = np.array([-24, 0.2, 0.2]).reshape(-1,1)
cost, grad = costFunction(test_theta, X, y) # Using part of X (after mapFeature)

print('Cost at test theta: %f\n' % cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(' %a \n' % grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X



''' ============= Part 3: Optimizing using fminunc  ============= '''
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.
y = y.flatten() 

def cost_reg(theta, X, y):
    J,_ = costFunction(theta, X, y, lambda_reg = 0)
    return J

def grad_reg(theta, X, y):
    _,grad = costFunction(theta, X, y, lambda_reg = 0)
    return grad.flatten()

from scipy.optimize import fmin_tnc

theta,_,_ = fmin_tnc(cost_reg, 
                x0 = initial_theta, 
                fprime = grad_reg, 
                args=(X, y),
                disp = False)

cost = cost_reg(theta, X,y)
theta = theta.reshape(-1,1)


#Print theta to screen
print('Cost at theta found by fminunc: %f\n'% cost)
print('Expected cost (approx): 0.203\n');
print('theta: \n')
print(' %a \n' % theta)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')



''' Plot Boundary '''
plot = plotDecisionBoundary(theta, X, y)
# Labels and Legend
plot.xlabel('Exam 1 score')
plot.ylabel('Exam 2 score')
plot.legend(['Admitted', 'Not admitted'])


plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')


''' ============== Part 4: Predict and Accuracies ============== '''
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m
#
#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

os.system('clear')  # For Linux/OS X

Xs = np.array([1, 45, 85]).reshape(1,-1)
prob = sigmoid(np.dot(Xs,theta))

print('For a student with scores 45 and 85, we predict\
an admission probability of %f\n' % prob)
print('Expected value: 0.775 +/- 0.002\n\n')      

# Compute accuracy on our training set
p = predict(theta, X)
p = p.flatten()
print('Train Accuracy: %f\n' % (np.mean(np.double(p == y))*100))
print('Expected accuracy (approx): 89.0\n')
print('\n')