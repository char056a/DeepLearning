
# activation functions
import numpy as np
def Relu(x):
    return (np.abs(x)+x)/2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
