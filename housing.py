# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:55:49 2020

@author: Carlos
"""
import numpy as np
import matplotlib.pyplot as plt
import linear_regresion as ln
import sklearn.model_selection as skl_ms


def read_data():
    array = np.genfromtxt("data/housing.csv",
                          dtype=float, delimiter=",", skip_header=1,
                          filling_values=0.,
                          converters={9: convert})

    labelsColum = 8

    # Delete labels columm
    X = np.delete(array, labelsColum, 1)
    
    
    X = ln.one_hot_encoding(X,8)

    # Delete string feature column
    #X = np.delete(X, 8, 1)

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

    alpha = 1
    num_iters = 1000

    theta, cost_history = ln.gradient_descent(X, y, alpha,
                                              num_iters)
    
    return theta, cost_history


def split_data(X, y):

    X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, y,
                                                               test_size=0.2)

    return X_train, X_test, y_train, y_test


def model_houseing():

    X, y = read_data()

    X = ln.normalize_features(X)
    
    # Add ones column
    X_new = np.c_[np.ones((len(X), 1)), X]

    X_train, X_test, y_train, y_test = split_data(X_new, y)
    
    # theta, cost_history = gradient_descent(X_train, y_train)
    
    # plt.plot(cost_history)
    
    # print(np.sqrt(cost_history[-1]))
    
    cs_train = np.empty(0)
    cs_test = np.empty(0)
    
    traning_examples = 400
    
    for i in range(1,traning_examples):
          theta, cost_history = gradient_descent(X_train[:i,:], y_train[:i,:])
          cs_train = np.append(cs_train,np.sqrt(cost_history[-1]))
          cs_test = np.append(cs_test, np.sqrt(ln.calculate_cost(X_test, theta,
                                                                y_test)))
          print(i)
    
    plt.plot(range(traning_examples-1),cs_train, label="train")
    plt.plot(range(traning_examples-1),cs_test, label="test")
    
    plt.legend()
 
    print("Error medio train: ", int(cs_train[-1]))
    print("Error medio Test: ",int(cs_test [-1]))

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

        grupo = np.append(grupo, np.array([[v, suma]]), 0)

        grupo = grupo[grupo[:,1].argsort(), :]
        
    y_pos = np.arange(len(grupo))
        
    plt.bar(y_pos, grupo[:, 1])
    
    plt.xticks(y_pos, grupo[:,0])


if __name__ == "__main__":
    # count_import()
    model_houseing()
