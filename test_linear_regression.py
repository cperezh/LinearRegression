# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:17:50 2020

@author: 30060126
"""
import numpy as np
import linear_regresion as ln

def test_predict():

    example = np.array([[1,2,3],[1,5,6]])
    parameters = np.array([[4,5,6]])
    
    p = ln.predict(example,parameters)
    
    if p[0] != 32 and p[1] != 65:
        raise Exception

def test_cost():
    test_set = np.array([[2,3],[4,5]])
    params = np.array([[0,1,2]])
    labels= np.array([[9],[15]])
    
    cost = ln.calculate_cost(test_set,params,labels)
    
    if cost!=0.5:
        raise Exception

def test_gradient_descent():
    
    test_set = np.array([[2,3],[4,5]])
    labels= np.array([[9],[15]])
    params = np.array([[0,0,0]])
    alpha = 0.1
    num_iters = 3
    
    ln.gradient_descent(test_set,labels,params,alpha,num_iters);

if (__name__=="__main__"):
    test_predict()
    test_cost()
    test_gradient_descent()
