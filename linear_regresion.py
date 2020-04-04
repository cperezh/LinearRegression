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

def test_cost():
    test_set = np.array([[2,3],[4,5]])
    params = np.array([[0,1,2]])
    labels= np.array([[9],[15]])
    
    cost = calculate_cost(test_set,params,labels)
    
    if cost!=0.5:
        raise Exception

def gradient_descent(X, y, theta, alpha, num_iters):
    
    m = len(X)
    
    for i in range(num_iters):
        
        predict = np.dot(X,theta.transpose())
        
        error = predict - y
        
        lern_rate = (alpha/m)*predict
                
        theta = theta - (np.dot(lern_rate,error))

def test_gradient_descent():
    gradient_descent(None,None,None,None,3);

def predict(X,theta):
    """
    Implementacio``n vectorial 

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

def test_predict():

    example = np.array([[1,2,3],[1,5,6]])
    parameters = np.array([[4,5,6]])
    
    p = predict(example,parameters)
    
    if p[0] != 32 and p[1] != 65:
        raise Exception
    
if __name__ == "__main__":
    test_predict()
    test_cost()
    test_gradient_descent()