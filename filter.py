import numpy as np

def minmod(r):
    return np.clip(r, 0, 1)

def superbee(r):
    return np.maximum(0, np.maximum(np.minimum(2*r, 1), np.minimum(r, 2)))

def vanleer(r):
    return (r+np.abs(r))/(1+np.abs(r))
