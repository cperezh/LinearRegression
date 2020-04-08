# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:18:57 2020

@author: 30060126
"""
import numpy as np

def file_2_Xy(path):
    array = np.loadtxt(path,delimiter = ",")
    
    X = array[:,0:np.size(array,1)-1];
    y = array[:,np.size(array,1)-1:]
    
    return X,y

if (__name__ == "__main__"):
    file_2_Xy("c:\prueba.csv")