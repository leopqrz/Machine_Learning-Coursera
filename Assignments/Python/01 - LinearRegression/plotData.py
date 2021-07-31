#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:35:35 2020

@author: leonardoqueiroz
"""

from matplotlib import pyplot as plt


def plotIterative(x,y,h):
    plt.subplot(121)
    plt.title('Iterative')
    plt.plot(x, y, 'rx')
    plt.plot(x, h,'b')

def plotConvergence(J_history):
    plt.subplot(122)
    plt.title('Convergence')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.plot(range(len(J_history)), J_history,'k')

def plotNormalEqn(x,y,h):
    plt.subplot(121)
    plt.title('Normal Equation')
    plt.plot(x, y, 'rx')
    plt.plot(x, h, 'g')

def plotSklearn(x,y,h):
    plt.subplot(122)
    plt.title('Sklearn')
    plt.plot(x, y,'rx')
    plt.plot(x, h.predict(x),'k')

