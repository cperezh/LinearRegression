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

    # Delete string feature column
    X = np.delete(X, 8, 1)

    # Read labels column
    y = array[:, labelsColum:labelsColum+1]

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


def plot_data(X, y):
    """
    Plots data in a scatter
    """

    fig, axs = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            axs[i][j].scatter(X[:, (i*3)+j:(i*3)+j+1].ravel(), y.ravel())


def gradient_descent(X, y):

    alpha = 0.1
    num_iters = 20000

    theta, cost_history = ln.gradient_descent(X, y, alpha,
                                              num_iters)

    plt.plot(cost_history)

    return theta, cost_history


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

    X = ln.normalize_features(X)

    # Add ones column
    X_new = np.c_[np.ones((len(X), 1)), X]

    # X, X_test, y, y_test = split_data(X, y)

    # plot_data(X, y)

    theta, cost_history = gradient_descent(X_new, y)

    # cost = ln.calculate_cost(X, theta, y)

    print("Error cuadratico medio: ", float(cost_history[-1]))
    print("Error medio: ", int(np.sqrt(cost_history[-1])))

    # print(cost)


def count_import():

    array = np.genfromtxt("data/housing.csv",
                          dtype=float, delimiter=",", skip_header=1,
                          filling_values=0.,
                          converters={9: convert})

    values = np.unique(array[:, 9])

    grupo = np.empty((0, 2))

    for i, v in enumerate(values[:]):

        solo_i = array[array[:, 9] == v, 8]

        suma = np.sum(solo_i) / len(solo_i)

        agg = np.array([[v, suma]])

        grupo = np.append(grupo, agg, 0)

        grupo = grupo[grupo[:,1].argsort(), :]

    plt.bar(grupo[:, 0], grupo[:, 1])


if __name__ == "__main__":
    count_import()
    # model_houseing()
