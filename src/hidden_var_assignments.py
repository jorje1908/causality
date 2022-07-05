import numpy as np
import scipy as sc
import pandas as pd

def circle_class(data, centers = None, r_small = 0.2, r_big = 0.5):
    """
    
    data: Dataframe with features x_0,...,x_d
    center = (list) with the centers of the circle function
    r_small: (float) small radius
    r_big: (float) big radius
    assigns data into clusters according to circle function
    
    """
    center = np.array(centers)
    if data.shape[1] != len(centers):
        raise ValueError('Data Should be in the form NxD and centers in the form [c1,...,cD]')
        
    ans = np.sqrt(np.sum((data.values-centers)**2, axis = 1))
    cls =  ((ans >= r_small) & (ans <= r_big)).astype(int)
    data['C'] = cls
    return data

def circle_class2(data, col_select = None, centers = None, r_small = None, r_big = None):
    #eligibility in circles
    centers = np.array(centers)
    cls = np.zeros(len(data)).astype(int)
    for c, rs, rb in zip(centers, r_small, r_big):
        ans = np.sqrt(np.sum((data.values-c)**2, axis = 1))
        cls = (cls |((ans >= rs) & (ans <= rb)))
        
    data['C'] = cls
    
    return data

def circle_class3(data, col_select = None, centers = None, r_small = None, r_big = None):
    
    #each circle its own eligibility
    centers = np.array(centers)
    cls = np.zeros(len(data)).astype(int)
    for i, (c, rs, rb) in enumerate(zip(centers, r_small, r_big)):
        ans = np.sqrt(np.sum((data.values-c)**2, axis = 1))
        cls[((ans >= rs) & (ans <= rb))] = i+1
        
    data['C'] = cls
    
    return data


def circle_class4(data, col_select = None, centers = None, eligibilities = None,  r_small = None, r_big = None):
    
    #each circle its own eligibility captured by eligibility list
    elg = eligibilities
    centers = np.array(centers)
    cls = np.zeros(len(data)).astype(int)
    for elig, c, rs, rb in zip(elg, centers, r_small, r_big):
        ans = np.sqrt(np.sum((data.values-c)**2, axis = 1))
        cls[((ans >= rs) & (ans <= rb))] = elig
        
    data['C'] = cls
    
    return data


