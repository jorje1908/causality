import numpy as np
import scipy as sc
import pandas as pd


def uniform_gen(N = 10000, D = 2, low = 0, high = 1):
    """
    N:(int) number of data points
    D:(int) number of dimensions
    low:(float) lowest range of generated points
    high:(float) highest range of generted points
    
    Generates a NxD matrix uniformly at random
    returns: DataFrame NxD with column names x_0, ..., x{D-1}
    """
    features = ['x'+str(i) for i in range(D)]
    
    X = np.random.uniform(low = low, high = high, size = (N,D))
    return pd.DataFrame(X, columns = features )