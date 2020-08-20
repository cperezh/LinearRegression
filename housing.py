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
    array = np.genfromtxt("data/housing_new.csv",
                          dtype=float, delimiter=",", skip_header=1,
                          filling_values=0.,
                          converters={9: convert_ocean_proximity})

    labelsColum = 8

    # Read labels column
    y = array[:, labelsColum:labelsColum+1]

    # Delete labels columm
    X = np.delete(array, labelsColum, 1)

    return X, y


def process_data(X, grado_poli_features=1):

    # One hot over ocean_proximity feature
    X = ln.one_hot_encoding(X, 8)

    poly = sk_pre.PolynomialFeatures(grado_poli_features)

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


def learn_model_houseing(X, y,alpha_val=1):
    '''
    Entrena el modelo sobre X para predecir Y

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    alpha_val : float
        regularization parameter

    Returns
    -------
    score_ridge_test : float
        score sobre el conjunto de test.

    '''

    X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, y,
                                                               test_size=0.2)

    #reg = sk_lnmodel.LinearRegression(normalize=True).fit(X_train, y_train)
    ridge = sk_lnmodel.Ridge(alpha = alpha_val,
                             normalize=True).fit(X_train, y_train)

    #score_ridge_train = ridge.score(X_train,y_train)
    score_ridge_test = ridge.score(X_test,y_test)

    #print("R2_ridge_train: ", score_ridge_train)
    #print("R2_ridge_test: ", score_ridge_test)

    #y_pred_train = ridge.predict(X_train)
    #y_pred_test = ridge.predict(X_test)

    #print("RMSE_train: ",
    #      sk_metrics.mean_squared_error(y_pred_train, y_train, squared=False))
    #print("RMSE_test: ",
    #      sk_metrics.mean_squared_error(y_pred_test, y_test, squared=False))

    #plt.scatter(range(y_test.shape[0]), y_test, c="b")
    #plt.scatter(range(y_test.shape[0]), y_pred_test, c="r")

    #print("Quartiles de error train: ",
    #      np.percentile(y_train-y_pred_train,
    #                    [0,10,25,50,75,90,100]).round(0))
    #print("Quartiles de error test: ",
    #      np.percentile(y_test-y_pred_test,
    #                   [0,10,25,50,75,90,100]).round(0))

    #joblib.dump(ridge, _MODEL_FILE)

    return score_ridge_test

def predict(X):

    reg = joblib.load(_MODEL_FILE)

    y_pred = reg.predict(X)

    return y_pred


def get_outliers(X):
    '''


    Parameters
    ----------
    X : np array de n x m
        array con ejemplos en las filas y features en las columnas.

    Returns
    -------
    indices_outliers : np array
        array de tamaño de filas de X con booleanos, donde True
        indica que esa fila de X es un outiler respecto a los
        valores de alguna de sus columnas.

    '''

    indices_outliers = np.full((X.shape[0]),False)

    iso_forest = sk_ens.IsolationForest(random_state=0)

    for feature_num in range(X.shape[1]):
        feature_rows = X[:, feature_num:feature_num+1]
        outliers = iso_forest.fit_predict(feature_rows)
        outliers_bool = outliers==-1
        indices_outliers = outliers_bool+indices_outliers
        num_out = np.count_nonzero(indices_outliers)

    return indices_outliers

def fit_degree_and_alpha():

    DEGREES = [1, 2, 3, 4, 5, 6, 7,  8]
    ALPHAS = [0.2, 0.5, 0.8, 1, 1.5, 2, 5, 10, 100]

    scores = np.empty((len(DEGREES),len(ALPHAS)))

    X, y = read_data()

    for i, degree in enumerate(DEGREES):

        print("Learning for degree ", degree)

        for j, alpha in enumerate(ALPHAS):

            print("Learning for alpha: ", alpha)

            X_processed = process_data(X,degree)

            score = learn_model_houseing(X_processed, y, alpha)

            scores[i,j] = score

            print("Score: ",score)

    print("SCORE: --->", scores)

    for i in range(scores.shape[0]):
        plt.plot(range(scores.shape[1]),scores[i,:])



def predict_main():
    X, y = read_data()

    X = process_data(X)

    y_pred = predict(X[1:2,:]).round(0)

    print(y_pred,y[1])


def build_new_file_without_outliers():

    X, y = read_data()

    outliers = get_outliers(X)

    outliers = get_outliers(y) + outliers

    with open("data/housing.csv", "r") as f:
        lines = f.readlines()

    with open("data/housing_new.csv", "w") as f_new:
        for i in range(len(lines)-1):
            if not outliers[i]:
                #el fichero de lineas tiene cabecera
                f_new.write(lines[i+1])


if __name__ == "__main__":

    fit_degree_and_alpha()
    #predict_main()
    #build_new_file_without_outliers()

