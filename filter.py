import numpy as np

def superbee(r):
    return np.maximum(0, np.maximum(np.minimum(2*r, 1), np.minimum(r, 2)))
