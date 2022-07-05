import numpy as np
import scipy as sc
import pandas as pd


def outcome1(data,  treatment = 'Treatment', cls = 'C', stats = None):
    """
    Produces the outcome Y as a function of Treatement and Class C
    data:DataFrame, should contain columns C and Treatment
    column_selection: list of columns to use for outcome
    stats: 4x2 matrix containing the means and standard deviations
     for the 4 cases 
     Case 1: Treatment 0 C 0
     Case 2: Treatment 0 C 1
     Case 3: Treatemnt 1 C 0
     Case 4: Treatemnt 1 C 1
    
    returns a dataframe like data but with two extra columns Y and Ycf
    """
    unique_eligibilities = len(data[cls].unique())
    ue = unique_eligibilities
    #print(ue)
    def sample(group):
        t = group[treatment].iloc[0]
        c = group[cls].iloc[0]
        tdot = 1-t
        statsf = stats[int(t*ue+c)]
        statscf = stats[int(tdot*ue+c)]
        N = len(group)
        group['Y'] = np.random.normal(loc = statsf[0], scale=statsf[1], size = N)
        group['Ycf'] = np.random.normal(loc = statscf[0], scale=statscf[1], size = N)
        
        return group
    
    new_data = data.groupby(by = [cls, treatment]).apply(sample).reset_index(drop = True)
    
    return new_data
