import numpy as np

def normal(dim_in,dim_out, mean=0.0, std=0.02):
    return np.random.normal(mean, std, size=(dim_out,dim_in))

def xavier_uniform(dim_in,dim_out):
    in_features, out_features = dim_out, dim_in
    lim = (6.0 / (in_features + out_features)) ** 0.5
    return np.random.uniform(-lim, lim, size=(dim_out,dim_in))

def xavier_normal(dim_in,dim_out):
    in_features, out_features = dim_out, dim_in
    std = (2.0 / (in_features + out_features)) ** 0.5
    return np.random.normal(0.0, std, size=(dim_out,dim_in))

def he_normal(dim_in,dim_out):
    in_features = dim_in
    std = (2.0 / in_features) ** 0.5
    return np.random.normal(0.0, std, size=(dim_out,dim_in))

def he_uniform(dim_in,dim_out):
    in_features = dim_in
    lim = (6.0 / in_features) ** 0.5
    return np.random.uniform(-lim, lim, size=(dim_out,dim_in))

def zeros(dim_in,dim_out):
    return np.zeros((dim_out,dim_in))

def random(dim_in,dim_out, low=-0.05, high=0.05):
    return np.random.uniform(low, high, size=(dim_out,dim_in))
