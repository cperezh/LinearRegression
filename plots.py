# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:16:49 2020

@author: Carlos
"""
import numpy as np
import matplotlib.pyplot as plt 
import utils as utl
import linear_regresion as ln

def plot_gradient_descent():
    X,y = utl.file_2_Xy("prueba.csv")
    theta = np.array([[1,1,1]])
    alpha = 0.000005
    num_iters = 2000000
    
    "Add cuadratic feature"
    X_new = ln.map_feature_2_cuadratic(X)
    
    "Add ones column"
    X_new = np.c_[np.ones((len(X),1)),X_new]
    
    
    fig, axs = plt.subplots(2)
    
    "PLOT Valores iniciales"
    axs[0].scatter(X.ravel(),y.ravel())
    
    "Calulo regression lineal"
    theta,cost_history = ln.gradient_descent(X_new,y,theta,alpha,num_iters)
    
    "PLOT Ajuste de coste"
    axs[1].plot(cost_history)
    
    "PLOT Prediccion"
    axs[0].plot(X.ravel(),ln.predict(X_new,theta).ravel())
    
    print("Prediccion: ",ln.predict(X_new,theta))
    
    print("Cost: ",cost_history[-1])

    
if __name__ == "__main__":
   plot_gradient_descent()