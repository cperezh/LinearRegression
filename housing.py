# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:55:49 2020

@author: Carlos
"""
import numpy as np
import matplotlib.pyplot as plt
import linear_regresion as ln
import sklearn.model_selection as skl_ms
import sklearn.preprocessing as sk_pre
import Normalizer as norma
from sklearn.ensemble import IsolationForest


def read_data():
    array = np.genfromtxt("data/housing.csv",
                          dtype=float, delimiter=",", skip_header=1,
                          filling_values=0.,
                          converters={9: convert_ocean_proximity})

    labelsColum = 8

    # Read labels column
    y = array[:, labelsColum:labelsColum+1]

    # Delete labels columm
    X = np.delete(array, labelsColum, 1)

    return X, y


def process_data(X):

    # One hot over ocean_proximity feature
    X = ln.one_hot_encoding(X, 8)

    # Build sintetic features
    #poly = sk_pre.PolynomialFeatures(1)

    #X = poly.fit_transform(X)

    normalizer = norma.Normalizer()

    X = normalizer.normalize_features(X)

    # X_new = X[:, [0,1,2,8,9,10,11,12,13]]
    # X_new = X[:, [0,1,2,8]]

    return X, normalizer


def convert_ocean_proximity(valor):
    """
    Mapeo del campo ocean_proximity de string a float
    """
    dict = {
        "NEAR BAY": 0,
        "<1H OCEAN": 1,
        "INLAND": 2,
        "NEAR OCEAN": 3,
        "ISLAND": 4
        }

    return dict[valor.decode()]


def plot_data_scatter(X, y):
    """
    Plots data in a scatter
    """

    fig, axs = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            axs[i][j].scatter(X[:, (i*3)+j:(i*3)+j+1].ravel(), y.ravel())


def plot_data(X, y):

    plt.scatter(X[:, [8]], y[:])


def box_plot_data(X, feature, subplot):
    subplot.boxplot(X[:, feature], showfliers=True)


def hist_data(X, feature, subplot):
    subplot.hist(X[:, feature])


def gradient_descent(X, y):

    alpha = 0.5
    num_iters = 10000

    theta, cost_history = ln.gradient_descent(X, y, alpha, num_iters)

    return theta, cost_history


def split_data(X, y):

    X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, y,
                                                               test_size=0.2)

    return X_train, X_test, y_train, y_test


def learn_model_houseing(X,y):

    X_train, X_test, y_train, y_test = split_data(X, y)

    theta, cost_history = gradient_descent(X_train, y_train)

    plt.plot(cost_history)

    print("Error train:", np.sqrt(cost_history[-1]))
    print("Error test:", np.sqrt(ln.calculate_cost(X_test, theta, y_test)))

    np.save("theta.npy",theta)

    # cs_train = np.empty(0)
    # cs_test = np.empty(0)

    # traning_examples = 3000

    # for i in range(1, traning_examples):
    #     theta, cost_history = gradient_descent(X_train[:i, :],
    #                                            y_train[:i, :])
    #     cs_train = np.append(cs_train, np.sqrt(cost_history[-1]))
    #     cs_test = np.append(cs_test, np.sqrt(ln.calculate_cost(X_test, theta,
    #                                                            y_test)))
    #     print(i)

    # plt.plot(range(traning_examples-1), cs_train, label="train")
    # plt.plot(range(traning_examples-1), cs_test, label="test")

    # plt.legend()

    # print("Error medio train: ", int(cs_train[-1]))
    # print("Error medio Test: ", int(cs_test[-1]))

    # print(cost)


def predict(X_new, y):

    theta = np.load("theta.npy")

    # X = normalizer.normalize_features(X, reuse=True)

    y_pred = ln.predict(X_new, theta)

    i = 2
    #plt.scatter(range(len(y[:i])), y[:i], c="b")
    #plt.scatter(range(len(y[:i])), y_pred[:i], c="r")

    cost = ln.calculate_cost(X_new[:i], theta, y[:i])

    print("coste: ", np.sqrt(cost))

    num_outliers = np.count_nonzero((y_pred[:, 0] - y[:, 0]) > 50000)
    cost = y_pred[:, 0] - y[:, 0]
    plt.boxplot(cost)
    print(np.percentile(cost,[0,10,25,50,75,90,100]))

    print("outliers: ",num_outliers)
    print("num examples: ",y_pred.shape[0])


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

        grupo = grupo[grupo[:, 1].argsort(), :]

    y_pos = np.arange(len(grupo))

    plt.bar(y_pos, grupo[:, 1])

    plt.xticks(y_pos, grupo[:, 0])


def outliers(X, feature_col, subplt):

    feature_rows = X[:, feature_col:feature_col+1]
    outliers = IsolationForest(random_state=0).fit_predict(feature_rows)

    salida = np.ndarray((feature_rows.shape[0], 2))

    salida[:, 0] = feature_rows[:, 0]
    salida[:, 1] = outliers[:]

    orden = salida[:,0].argsort();

    salida = salida[orden]

    subplt.plot(salida[:, 0],salida[:, 1])

    return salida

def delete_outliers(X, y):

    for feature_num in range(X.shape[1]):
        feature_rows = X[:, feature_num:feature_num+1]
        outliers = IsolationForest(random_state=0).fit_predict(feature_rows)
        indices_outliers = outliers==-1
        num_out = np.count_nonzero(indices_outliers)
        X = np.delete(X,indices_outliers,0)
        y = np.delete(y,indices_outliers,0)

    return X,y

if __name__ == "__main__":

    X, y = read_data()

    X_without_outliers,y_without_outliers = delete_outliers(X,y)

    X_new, normalizer = process_data(X_without_outliers)


    # plot_data_scatter(X, y)
    # plot_data(X, y)

    #feature_num = 7

    #fig, axs = plt.subplots(2, 1)
    #box_plot_data(X_without_outliers, feature_num, axs[0])
    #hist_data(X_without_outliers, feature_num, axs[1])

    #learn_model_houseing(X_new, y_without_outliers)
    predict(X_new, y_without_outliers)
