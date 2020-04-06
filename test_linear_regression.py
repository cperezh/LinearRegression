# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:17:50 2020

@author: 30060126
"""
import numpy as np
import linear_regresion as ln
import matplotlib.pyplot as plt 

def test_predict():

    X = np.array([[2,3],[5,6]])
    theta = np.array([[4,5,6]])
    
    X = np.c_[np.ones((len(X),1)),X]
    
    p = ln.predict(X,theta)
    
    if p[0] != 32 and p[1] != 65:
        raise Exception

def test_cost():
    X = np.array([[2,3],[4,5]])
    theta = np.array([[0,1,2]])
    y = np.array([[9],[15]])
    
    X = np.c_[np.ones((len(X),1)),X]
    
    cost = ln.calculate_cost(X,theta,y)
    
    if cost!=0.5:
        raise Exception

def test_gradient_descent():
    
    X = np.array([[2],[4],[3],[10]])
    y= np.array([[9],[15],[10],[30]])
    theta = np.array([[1,1]])
    alpha = 0.0001
    num_iters = 5000
    
    X_new = np.c_[np.ones((len(X),1)),X]
    
    fig, axs = plt.subplots(2)
    
    axs[0].scatter(X.ravel(),y.ravel())
    
    theta,cost_history = ln.gradient_descent(X_new,y,theta,alpha,num_iters)
    
    print(cost_history)
    
    axs[1].plot(cost_history)
    
    axs[0].plot(X.ravel(),ln.predict(X_new,theta).ravel())
    
    print("Prediccion: ",ln.predict(X_new,theta))
    

if (__name__=="__main__"):
    test_predict()
    test_cost()
    test_gradient_descent()
