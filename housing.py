# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:55:49 2020

@author: Carlos
"""
import numpy as np
import matplotlib.pyplot as plt
import linear_regresion as ln


def read_data():
    array = np.genfromtxt("data/housing.csv",
                          dtype=float, delimiter=",", skip_header=1,
                          filling_values=0.,
                          converters={9: convert})

    labelsColum = 8

    # Delete labels columm
    X = np.delete(array, labelsColum, 1)

    # Read labels column
    y = array[:, labelsColum:labelsColum+1]

    # validar(X, y)

    # Convert to float
    X = X.astype(np.float)
    y = y.astype(np.float)
    
    # Add ones column
    X_new = np.c_[np.ones((len(X), 1)), X]

    return X_new, y


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
            print("b: ", b)
            print("Error: ", err)
            raise err


def plot_data(X, y):

    fig, axs = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            axs[i][j].scatter(X[:, (i*3)+j:(i*3)+j+1].ravel(), y.ravel())


def gradient_descent(X, y):

    theta = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    alpha = 0.0000001
    num_iters = 30000

    theta, cost_history = ln.gradient_descent(X, y, theta, alpha,
                                              num_iters)

    plt.plot(cost_history)
    
    return  theta, cost_history


def split_data(X, y):
    
    total_rowa = len(X)
    learning_rows = int(total_rowa * 0.8)
    test_rows = int(total_rowa * 0.2)
    
    X_learning = X[0:learning_rows, :]
    X_test = X[learning_rows:, :]
    y_learning = y[0:learning_rows, :]
    y_test = y[learning_rows:, :]
    
    return X_learning, X_test, y_learning, y_test

def model_houseing():

    X, y = read_data()

    X, X_test, y, y_test = split_data(X, y)

    # plot_data(X, y)

    theta, cost_history = gradient_descent(X, y)
    
    cost = ln.calculate_cost(X_test, theta, y_test)
    
    print(float(cost_history[-1]))
    
    print(cost)


if __name__ == "__main__":
    model_houseing()
