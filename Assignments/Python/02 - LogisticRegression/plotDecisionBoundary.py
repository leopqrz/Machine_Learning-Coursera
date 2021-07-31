#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 03:37:36 2020

@author: leonardoqueiroz
"""
import numpy as np
from plotData import plotData
from mapFeature import mapFeature


def plotDecisionBoundary(theta, X, y, degree = 6):
    
    plt = plotData(X[:,1:3], y)
    
    if np.size(X, 1) <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X[:,1])-2,  max(X[:,1])+2]
    
        # Calculate the decision boundary line
        plot_y = -1/theta[2] * (theta[1] * plot_x + theta[0])
    
        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)
        
        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
    
        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                Z = np.array((u[i], v[j])).reshape(1,-1)
                z[i, j] = np.dot(mapFeature(Z, degree), theta)
                
        z = z.T # important to transpose z before calling contour
    
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, 0)
    
    return plt
