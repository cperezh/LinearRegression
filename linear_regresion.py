import numpy as np

def calculate_cost(X,theta,y):
    """
    Calcula el error cuadratico medio de X respecto de Y
    utilizando parametros theta en una Regresion Lineal

    Parameters
    ----------
    X : np.array m*n 
        matrix of n elements to predict with m features.
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
    
    X = np.c_[np.ones((m,1)),X]

    a = np.dot(X,theta.transpose())-y
    c = a.transpose()
    
    cost = (1/(2*m))*np.dot(c,a)
  
    return cost



def gradient_descent(X, y, theta, alpha, num_iters):
    """
    X: Examples
    y: labels
    """        
    
    m = len(X)
    
    X = np.c_[np.ones((m,1)),X]
    
    for i in range(num_iters):
        
        pred= predict(X,theta)
        
        error = pred - y
        
        med = alpha / m
        
        lern_rate = med*pred
                
        theta = theta - (np.dot(lern_rate.transpose(),error))



def predict(X,theta):
    """
    Implementacion vectorial 

    Parameters
    ----------
    X : np.array n*m dim
        matrix of n elements to predict with m features
    theta : np.array 1xm dim
        row matrix of linear regresion perarameters
    Returns
    -------
    np.array n*1 dim
        matrix with the prediction form n elements.

    """
    
    result = np.dot(X,theta.transpose())
    
    return result


    
if __name__ == "__main__":
   pass