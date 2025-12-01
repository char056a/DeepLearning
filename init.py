import numpy as np


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

