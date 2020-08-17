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
import sklearn.metrics as sk_metrics
import Normalizer as norma
import joblib
import sklearn.ensemble as sk_ens
import sklearn.linear_model as sk_lnmodel

_MODEL_FILE = "./reg.joblib"


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

    poly = sk_pre.PolynomialFeatures(8)

    X = poly.fit_transform(X)

    return X


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


def learn_model_houseing(X, y):

    X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, y,
                                                               test_size=0.2)

    #reg = sk_lnmodel.LinearRegression(normalize=True).fit(X_train, y_train)
    ridge = sk_lnmodel.Ridge(normalize=True, alpha=1000).fit(X_train, y_train)

    score = reg.score(X_test,y_test)

    score_ridge = ridge.score(X_test,y_test)


    print("R2: ", score)

    print("R2_ridge: ", score_ridge)

    y_pred = reg.predict(X_test)

    print("RMSE: ",
          sk_metrics.mean_squared_error(y_test, y_pred,squared=False))

    plt.scatter(range(y_test.shape[0]), y_test, c="b")
    plt.scatter(range(y_test.shape[0]), y_pred, c="r")

    print("Quartiles de error: ",
          np.percentile(y_test-y_pred, [0,10,25,50,75,90,100]).round(0))

    joblib.dump(reg, _MODEL_FILE)



def predict(X):

    reg = joblib.load(_MODEL_FILE)

    y_pred = reg.predict(X)

    return y_pred


def get_outliers(X):

    indices_outliers = np.full((X.shape[0]),False)

    iso_forest = sk_ens.IsolationForest(random_state=0)

    for feature_num in range(X.shape[1]):
        feature_rows = X[:, feature_num:feature_num+1]
        outliers = iso_forest.fit_predict(feature_rows)
        outliers_bool = outliers==-1
        indices_outliers = outliers_bool+indices_outliers
        num_out = np.count_nonzero(indices_outliers)

    return indices_outliers

def learn_main():
    X, y = read_data()

    outliers = get_outliers(X)

    outliers = get_outliers(y) + outliers

    X_without_outliers = np.delete(X, outliers, 0)

    y_without_outliers = np.delete(y, outliers, 0)

    X_without_outliers = process_data(X_without_outliers)

    learn_model_houseing(X_without_outliers, y_without_outliers)


def predict_main():
    X, y = read_data()

    X = process_data(X)

    y_pred = predict(X[1:2,:]).round(0)

    print(y_pred,y[1])


if __name__ == "__main__":

    learn_main()
    #predict_main()

