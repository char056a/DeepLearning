

# activation functions and their derivatives
import numpy as np
def Relu(x,derivative=False):

    if derivative:
        return 1*(x>0)
    else:
        return (np.abs(x)+x)/2

def tanh(x,derivative = False):

    if derivative: 
        return 1-np.tanh(x)**2
    else:
        return np.tanh(x)
   


