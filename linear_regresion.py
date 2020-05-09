import numpy as np
import sklearn.preprocessing as skl_pre

"""
TODO
    1. Hacer la predicción con normalizacion
    2. Hace scatter pediccion vs real
    3. Hacer API para llamar a la predicción
    4. Utilizar otras librerias de SCIKIT LEARN
"""


def calculate_cost(X, theta, y):
    """
    Calcula el error cuadratico medio de X respecto de Y
    utilizando parametros theta en una Regresion Lineal

    Parameters
    ----------
    X : np.array m*n+1
        matrix of m elements to predict with n features. The first
        colum must be ones columns
    theta : np.array 1*(n+1)
        row matrix of linear regresion perarameters.
    Y : np.array m*1
        matrix of m elements with the labels of X examples

    Returns
    -------
    eror_cuadra : number
        Devuelve el error cuadratico medio.

        cost = (1/(2*m))*np.sum((pred - Y)**2)

    """

    m = len(X)
    a = np.dot(X, theta.transpose())-y
    c = a.transpose()

    cost = (1/(2*m))*np.dot(c, a)

    return cost


def gradient_descent(X, y, alpha, num_iters):
    """
    X: Examples np.ndarray
    y: labels
    theta: parameters  np.array 1*n+1 dim
    alpha: learning rate
    num_iters: iterations of gradient descent

    return
    ------
    theta learned parameters
    cost_histoy an np.array of length num_iters with the cost of
    every iteration
    """

    theta = initialize_theta(X.shape[1])

    m = len(X)

    cost_history = np.zeros(num_iters)

    for i in range(num_iters):

        pred = predict(X, theta)

        error = pred - y

        temp = np.dot(error.transpose(), X)

        theta = theta - ((alpha/m)*temp)

        cost_history[i] = calculate_cost(X, theta, y)

    return (theta, cost_history)


def initialize_theta(size):
    """
    Parameters
    ----------
    size : int
        Numero de features, incluyendo x0.

    Returns
    -------
    theta : np.ndarray
        Thete parameters with random iniit. 1*size matrix

    """

    theta = np.empty((1, size))

    rng = np.random.default_rng()

    for i in range(size):
        vals = rng.standard_normal()
        theta[0][i] = vals

    return theta


def predict(X, theta):
    """
    Implementacion vectorial

    Parameters
    ----------
    X : np.array m*n dim
        matrix of m elements to predict with n features
    theta : np.array 1*n+1 dim
        row matrix of linear regresion perarameters
    Returns
    -------
    np.array n*1 dim
        matrix with the prediction form n elements.

    """

    result = np.dot(X, theta.transpose())

    return result


def one_hot_encoding(X, feature_column):

    data = X[:, feature_column:feature_column+1]

    unique_cat = np.unique(data)

    categories = np.reshape(unique_cat, (len(unique_cat), 1))

    enccoder = skl_pre.OneHotEncoder()

    enccoder.fit(categories)

    new_features = enccoder.transform(data).toarray()

    # Selecciono las columnas que no sean la de la caracteristica
    new_colums = np.arange(X.shape[1])[:] != feature_column

    X = X[:, new_colums]

    X = np.concatenate((X, new_features), 1)

    return X


if __name__ == "__main__":

    X = np.array([[1., 2., 3.],[4., 5., 6.]])

    normalize_features(X)