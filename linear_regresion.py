import numpy as np


def calculate_cost(X,theta,y):
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
    a = np.dot(X,theta.transpose())-y
    c = a.transpose()
    
    cost = (1/(2*m))*np.dot(c,a)
  
    return cost



def gradient_descent(X, y, theta, alpha, num_iters):
    """
    X: Examples
    y: labels
    theta: parameters
    alpha: learning rate
    num_iters: iterations of gradient descent
    
    return
    ------
    cost_histoy an np.array of length num_iters with the cost of every iteration
    """        
    
    m = len(X)
    
    cost_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        
        pred= predict(X,theta)
        
        error = pred - y
        
        temp = np.dot(error.transpose(),X)
        
        theta = theta - ((alpha/m)*temp)
        
        cost_history[i] = calculate_cost(X,theta,y)
        
    return (theta,cost_history)


def predict(X,theta):
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
    
    result = np.dot(X,theta.transpose())
    
    return result

def map_feature_2_cuadratic(X):
    """
    

    Parameters
    ----------
    X : nparray 
        mapea una matriz de m*1 a m*2. Convierte una feature
        lineal en cuadratica

    Returns
    -------
    nparray m*2.

    """
    "Si hay mas de un feature"
    if np.size(X,1) != 1:
        raise Exception
    
    X = np.c_[X,X**2]
    
    return X
    
