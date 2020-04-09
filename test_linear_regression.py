# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:17:50 2020

@author: 30060126
"""
import numpy as np
import linear_regresion as ln


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
    
    theta,cost_history = ln.gradient_descent(X_new,y,theta,alpha,num_iters)
    
    if cost_history[-1] != 0.7889417649835131:
        raise Exception

def test_map_feature_2_cuadratic():
    X = np.array([[1],[2],[3]])
    print(ln.map_feature_2_cuadratic(X))
    
if (__name__=="__main__"):
    test_predict()
    test_cost()
    test_gradient_descent()
    test_map_feature_2_cuadratic()
