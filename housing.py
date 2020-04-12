# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:55:49 2020

@author: Carlos
"""
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    array = np.genfromtxt("data/housing.csv",
                       dtype=float, delimiter=",",  skip_header=1, 
                       filling_values=0.,
                       converters = {9: convert})

    labelsColum = 8

    # Delete text feature
    X = np.delete(array, 9, 1)

    # Delete labels columm
    X = np.delete(X, labelsColum, 1)

    # Read labels column
    y = array[:, labelsColum:labelsColum+1]

    #validar(X, y)

    # Convert to float
    X = X.astype(np.float)
    y = y.astype(np.float)

    return X, y


def convert(valor):
    """
    Mapeo del campo string a float
    """
    dict = {
        "NEAR BAY": 0,
        "<1H OCEAN": 1,
        "INLAND": 2,
        "NEAR OCEAN": 3,
        "ISLAND": 4
        }

    return dict[valor.decode()]


def validar(X, y):
    for i in range(len(X)):
        print(i, " >>>> ", X[i])
        try:
            if (i == 290):
                print(ord(X[i][4]))
            b = X[i].astype(np.float)
        except Exception as err:
            print("b: ",b)
            print("Error: ", err)
            raise err
        

def plot_data(X, y):
    first_feature = X[:, 0:1]
    plt.boxplot(first_feature.ravel())


def model_houseing():
    X, y = read_data()

    plot_data(X, y)


if __name__ == "__main__":
    model_houseing()
