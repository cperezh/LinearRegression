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

def plot_gradient_descent():
    X = np.array([[2],[4],[3],[10]])
    y= np.array([[9],[15],[10],[30]])
    theta = np.array([[1,1]])
    alpha = 0.0001
    num_iters = 5000
    
    X_new = np.c_[np.ones((len(X),1)),X]
    
    fig, axs = plt.subplots(2)
    
    axs[0].scatter(X.ravel(),y.ravel())
    
    theta,cost_history = ln.gradient_descent(X_new,y,theta,alpha,num_iters)
    
    #for i in range(num_iters):
    axs[1].plot(cost_history)
    axs[0].plot(X.ravel(),ln.predict(X_new,theta).ravel())
    
    print("Prediccion: ",ln.predict(X_new,theta))

    
if __name__ == "__main__":
   plot_gradient_descent()