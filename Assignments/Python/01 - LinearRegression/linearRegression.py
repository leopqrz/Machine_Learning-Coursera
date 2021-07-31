#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:08:30 2020

@author: leonardoqueiroz
"""


''' Machine Learning Online Class - Exercise 1: Linear Regression

  Instructions
  ------------

  This file contains code that helps you get started on the
  linear exercise. You will need to complete the following functions
  in this exericse:

     plotData.py
     gradientDescent.py
     computeCost.py
     gradientDescentMulti.py
     computeCostMulti.py
     featureNormalize.py
     normalEqn.py

  For this exercise, you will not need to change any code in this file,
  or any other files other than those mentioned above.

 x refers to the population size in 10,000s
 y refers to the profit in $10,000s
'''


from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn import linear_model

from mapFeature import mapFeature
from featureNormalize import featureNormalize
from costFunction import costFunction
from gradientDescent import gradientDescent
from plotData import plotConvergence, plotIterative, plotNormalEqn, plotSklearn
from normalEqn import normalEqn

# Some Gradiente descent settinngs
order = 1 # Doesn't work for more features
lamb = 0 # Regularization
iterations = 1500; # 1500: 1 feature, 400: 2 features
alpha = 0.01;
normalize = False


''' Part 1: Plotting  '''

os.system('clear')  # For Linux/OS X

data = np.loadtxt("ex1data1.txt", dtype='f', delimiter=',')

x = data[:,0].reshape(-1,1)
X = np.copy(x)
y = data[:,1].reshape(-1,1)
m = len(y)

''' Prediction for tests '''
h = linear_model.LinearRegression()    
h.fit(x, y)



''' Normalizing Features '''
if normalize:
    X, mu, sigma = featureNormalize(X);


''' Part 2: Cost and Gradient Descent '''
# Add intercept term to X
    
X = mapFeature(X, order = 1)        

print('Testing the Cost Function...\n')
theta = np.zeros((X.shape[1],1)) # Start t0 and t1 (Parameters)   
J = costFunction(X, y, theta, lamb);
print('With theta = [0 ; 0]\n\nCost computed = %f\n' % J[0][0]);
cost = lambda theta,X,y: (1/(2*len(y))) * sum((np.dot(X,theta) - y)**2)
print('Expected cost value (approx) %f\n' % cost(theta,X,y))

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X
plt.pause(0.01)
print('Testing the Cost Function...\n')
theta_ = np.random.rand(X.shape[1],1)# Start t0 and t1 (Parameters)   
J = costFunction(X, y, theta_, lamb);
print('With theta = \n%a\n\nCost computed = %f\n' % (theta_, J[0][0]));
print('Expected cost value (approx) %f\n' % cost(theta_,X,y))

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X
print('Running Gradient Descent ...\n')
# run gradient descent
theta1, J_history = gradientDescent(X, y, theta, alpha, iterations, lamb);
# print theta to screen
print('Theta found by gradient descent:\n')
print(theta1)
print('Expected theta values (approx)\n')
print(' %f\n  %f\n\n' % (h.intercept_, h.coef_))

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X
print('Plotting Linear Regression and Convergence')
# Plotting the graphics
plt.figure(1)
plt.figure(figsize=(10,5))

h = np.dot(X,theta1)
plotIterative(x, y, h)
plotConvergence(J_history)


plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X
print(' Visualization J(theta_0, theta_1) ... ')

''' Part 3: Visualization J(theta_0, theta_1) ''' 
# Grid over which we will calculate J

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
theta0_valsX, theta1_valsY = np.meshgrid(theta0_vals, theta1_vals)

thetas = np.array((np.ravel(theta0_valsX), np.ravel(theta1_valsY))).T.reshape(-1,2,1)

J_valsZ = np.array([costFunction(X,y,t,lamb) for t in thetas])
J_valsZ = J_valsZ.reshape(theta0_valsX.shape)

ax.plot_surface(theta0_valsX, theta1_valsY, J_valsZ)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'$J(\theta_0, \theta_1)$')

plt.show()


plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X
print(' Visualization J(theta_0, theta_1) contour ... ')


# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
fig = plt.figure(figsize=(8,8))

plt.contour(theta0_vals, theta1_vals, J_valsZ, np.logspace(-2, 3, 20))
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(theta1[0], theta1[1], 'rx')

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

os.system('clear')  # For Linux/OS X
print('Plotting Normal Equation and Sklearn Linear Regression')

plt.figure(2)
plt.figure(figsize=(10,5))

''' Linear Regression - Normal Equation '''
theta2 = normalEqn(X,y,lamb)
h = np.dot(X,theta2)
plotNormalEqn(x,y,h)

''' Linear Regression - sklearn '''

h = linear_model.LinearRegression()    
h.fit(x, y)
plotSklearn(x,y,h)
theta3 = np.array((h.intercept_, h.coef_))

plt.pause(0.01)
input('\n\n\n\nPRESS ENTER...\n')

''' Part 4: Prediction Test '''
os.system('clear')  # For Linux/OS X
print("Test Prediction: Interactive, Normal Quation and Sklearn \n")

xx = np.array([1, 70000])    
predict1 = np.dot(theta1.T, xx)
predict2 = np.dot(theta2.T, xx)
predict3 = np.dot(theta3.T, xx)
print('\npredict1: %f \npredict2: %f \npredict3: %f' % (predict1, predict2, predict3))



