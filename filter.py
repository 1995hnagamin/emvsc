import numpy as np

def superbee(r):
    return np.max([0, np.min([2*r, 1]), np.min(r, 2)])
