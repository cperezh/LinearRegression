# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:11:14 2020

@author: Carlos
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    a = np.array([[1,2],[2,3],[3,1],[2,4]])
    
    b = np.array([10,70,20])
    
    plt.pie(b)
    #plt.bar(a[:,1],a[:,0],color="blue")