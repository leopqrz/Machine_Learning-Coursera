#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 03:01:58 2020

@author: leonardoqueiroz
"""
import numpy as np
from scipy.optimize import fmin_cg
from lrCostFunction import lrCostFunction




def cost_reg(theta, X, y, lambda_reg):
    J,_ = lrCostFunction(theta, X, y, lambda_reg)
    return J

def grad_reg(theta, X, y, lambda_reg):
    _,grad = lrCostFunction(theta, X, y, lambda_reg)
    return grad.flatten()


def oneVsAll(X, y, num_labels, lambda_reg):

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m,1)), X))

    initial_theta = np.zeros((n + 1, 1))


    # change the dimension from (m,1) to (m,) for minimization
    y = y.flatten() 
    initial_theta = initial_theta.flatten()

    for c in range(1, num_labels+1):
 
        print('Training %d out of %d categories...' % (c, num_labels))
        y_ = np.double(y==c)
        
        theta = fmin_cg(cost_reg, 
                        x0 = initial_theta, 
                        fprime = grad_reg, 
                        args=(X, y_, lambda_reg),
                        maxiter = 50, disp = False)

        all_theta[c-1,:] = theta

    return all_theta