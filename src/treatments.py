import numpy as np
import scipy as sc
import pandas as pd



def uniform_treat(data, choices = None, probabilities = None):
    """
    data:(pandas array) columns x1,...,xD, C
    choices: list with integers to sample from
    probabilities: probabilities to sample each integer from choices
    (need to sum to 1)
    Generates treatment assignment uniformly at random
    """
    N = len(data)
    ch = np.random.choice(choices, size=N, replace=True, p=probabilities)
    data['Treatment'] = ch
    return data