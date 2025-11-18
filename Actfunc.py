

# activation functions and their derivatives
import numpy as np
def Relu(x,derivative=False):

    if derivative:
        return 1*(x>0)
    else:
        return (np.abs(x)+x)/2

def sigmoid(x,derivative=False):

    if derivative:
        return np.exp(-x)/(1+2*np.exp(-x)+np.exp(-2*x))
    else:
        return 1/(1+np.exp(-x))

def tanh(x,derivative = False):

    if derivative: 
        return 1-np.tanh(x)**2
    else:
        return np.tanh(x)
   


