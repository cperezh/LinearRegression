# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 11:26:19 2020

@author: Carlos
"""

import numpy as np

if __name__ == "__main__":
    a = np.ndarray((3,2))
    a = [[1,2],[3,4],[5,6]]
    print(a)
    print(type(a))
    a = np.array([[1,2],[3,4],[5,6]])
    print(a)
    print(type(a))
    a = np.delete(a,[True,False,False],0)
    print(a)