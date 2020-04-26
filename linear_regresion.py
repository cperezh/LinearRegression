import numpy as np
import sklearn.preprocessing as skl_pre

"""
TODO 
1. Normalizar las features
2. Implementar regresion polinomial
3. Ver gráfca del Test set en funcion del training set size
"""


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
    cost_histoy an np.array of length num_iters with the cost of every iteration
    """        
    
    theta = initialize_theta(X.shape[1])
    
    m = len(X)
    
    cost_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        
        pred= predict(X,theta)
        
        error = pred - y
        
        temp = np.dot(error.transpose(),X)
        
        theta = theta - ((alpha/m)*temp)
        
        cost_history[i] = calculate_cost(X,theta,y)
        
    return (theta,cost_history)


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
    
    
    theta = np.empty((1,size))
    
    rng = np.random.default_rng()
    
    for i in range(size):
        vals = rng.standard_normal()
        theta[0][i] = vals
        
    return theta
    

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
        
    primera_columna = X[:,[0]]
    
   # X = np.c_[X,primera_columna**2]
    
   # X = np.c_[X,primera_columna**3]
    
    X = np.c_[X,np.exp(primera_columna)]
    
    return X


def map_polinomial_features(X,list_of_features):
    
    X_temp = X[:,list_of_features]
    
    for feature in list_of_features:
        new_feat_cuadratic = X[:,feature:feature+1]**2
        new_feat_cubic = X[:,feature:feature+1]**2
        new_feat_exp = np.exp(X[:,feature:feature+1])
        X = np.concatenate((X,new_feat_cuadratic,new_feat_cubic),1)
    
    return X


def normalize_features(X):
    """
    Normilize features matrix with mean normalization
    and feature (max-min) acaling

    Parameters
    ----------
    X : np.ndarray
        n*m+1 features matrix

    Returns
    -------
    X : np.ndarray
        n*m+1 features matrix normalized

    """
 
    # For every feature
    for i in range(X.shape[1]):
        
        mean = np.mean(X[:,i])
        
        x_range = X[:,i].max() - X[:,i].min()
        
        X[:,i] = (X[:,i] - mean )/ x_range
    
    return X


def one_hot_encoding(X,feature_column):
    
    data = X[:,feature_column:feature_column+1]
    
    unique_cat = np.unique(data)
    
    categories = np.reshape(unique_cat,(len(unique_cat),1))
    
    enccoder = skl_pre.OneHotEncoder()
    
    enccoder.fit(categories)
    
    new_features = enccoder.transform(data).toarray()
    
    # Selecciono las columnas que no sean la de la caracteristica
    new_colums = np.arange(X.shape[1])[:] != feature_column
    
    X = X[:,new_colums]
    
    X = np.concatenate((X,new_features),1)
    
    return X

if __name__ == "__main__":
    
    X = np.array([[1.,2.,3.],[4.,5.,6.]])
    
    normalize_features(X)