
# activation functions and their derivatives
import numpy as np
def Relu(x):
    return (np.abs(x)+x)/2

def d_Relu(x):
    return (np.abs(x)+x)/(2*np.abs(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return (1/(1+np.exp(-x))) *(1-(1/(1+np.exp(-x))))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def d_tanh(x):
    return 1-tanh(x)**2
