# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:55:49 2020

@author: Carlos
"""
import numpy as np


def readData():
    array = np.loadtxt("data/housing.csv",
                       delimiter=",", dtype=str, skiprows=1)

    labelsColum = 8

    # Delete text feature
    X = np.delete(array, 9, 1)

    # Delete labels columm
    X = np.delete(X, labelsColum, 1)

    # Read labels column
    y = array[:, labelsColum:labelsColum+1]

    return X, y


def model_houseing():
    X, y = readData()


if __name__ == "__main__":
    model_houseing()
