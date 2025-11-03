import numpy as np

def normal(shape, mean=0.0, std=0.02):
    return np.random.normal(mean, std, size=shape)

def xavier_uniform(shape):
    in_features, out_features = shape[1], shape[0]
    lim = (6.0 / (in_features + out_features)) ** 0.5
    return np.random.uniform(-lim, lim, size=shape)

def xavier_normal(shape):
    in_features, out_features = shape[1], shape[0]
    std = (2.0 / (in_features + out_features)) ** 0.5
    return np.random.normal(0.0, std, size=shape)

def he_normal(shape):
    in_features = shape[1]
    std = (2.0 / in_features) ** 0.5
    return np.random.normal(0.0, std, size=shape)

def he_uniform(shape):
    in_features = shape[1]
    lim = (6.0 / in_features) ** 0.5
    return np.random.uniform(-lim, lim, size=shape)

def zeros(shape):
    return np.zeros(shape)

def random(shape, low=-0.05, high=0.05):
    return np.random.uniform(low, high, size=shape)
