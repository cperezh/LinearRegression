# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:53:56 2020

@author: Carlos
"""
import numpy as np


def convert(valor):
    dict = {"a":1,"b":2}
    print(">>>>>>",valor," ",type(valor)," ",valor.decode())
    return dict[valor.decode()]
    

if __name__=="__main__":
    datos = np.genfromtxt("data/housing_test2.csv",
                       dtype=float, delimiter=",",  
                       converters = {3: convert}, 
                       filling_values=0.)