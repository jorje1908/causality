import numpy as np
import pandas as pd
import scipy as sc

from distributions import*
from hidden_var_assignments import*
from outcomes import*
from treatments import*




class generate_experiment():
    """
    base class for experiment generation
    it includes 1 Steps
    1. Generate Features
    2. Assign them into different classes
    3. Assign treatment
    4. Assign Outcome.
    """
    
    
    
    def __init__(self, kwargs_generate, kwargs_assign_class, 
                 kwargs_assign_treatment, kwargs_assign_outcome):
        
       # print('Generating Experiment')
        data = self.generate_data_points(**kwargs_generate)
        data = self.assign_class(data.copy(), **kwargs_assign_class)
        data = self.assign_treatment(data.copy(), **kwargs_assign_treatment)
        data = self.assign_outcome(data.copy(), **kwargs_assign_outcome)
        self.dat = data
        
        return 
        
    def generate_data_points(self, N = 10000, D = 2, f_gen_name = 'uniform_dist', **kwargs_fun):
        """
        N:(int) Number of points to genrate
        D:(int) Number of dimensions 
        f_gen_name:(string) Name of distribution to generate points
        kwargs_fun: dictionary 

        """

    
        return eval(f_gen_name+'(N, D, **kwargs_fun)')

    def assign_class(self, data, f_class_name = 'circle_class',skip = False , **kwargs_fun):
        """
        data: Dataframe with features
        f_class_name: the name of the function to be applied to 
        generate the clusters
        kwargs_fun: keyword arguments to go into the generation function

        returns: DataFrame same as data but with an extra column C
        containing the class each point belongs
        """
        if skip:
            print('Skipping class assignment')
            return data
        return eval(f_class_name+'(data,**kwargs_fun)')
    
    
    #3.
    def assign_treatment(self, data, f_treat_name = 'uniform_treat', skip = False, **kwargs_fun):
        """
        data: Dataframe with features and class C
        f_treat_name: the name of the function to be applied to 
        generate the treatment groups
        kwargs_fun: keyword arguments to go into the treatment generation function

        returns: DataFrame same as data but with an extra column T
        containing the treatemnt group each point belongs
        """
        if skip:
            print('Skipping treatment assignment')
            return data
        return eval(f_treat_name+'(data,**kwargs_fun)')
    
    #4.
    def assign_outcome(self, data, f_outcome_name = 'outcome1',skip = False, **kwargs_fun):
        """
        data: Dataframe with features, class C and treatment "Treatment"
        f_treat_name: the name of the function to be applied to 
        generate the treatment groups
        kwargs_fun: keyword arguments to go into the treatment generation function

        returns: DataFrame same as data but with an extra column T
        containing the treatemnt group each point belongs
        """
        if skip:
            print('Skipping Outcome Assignment')
            return data
        return eval(f_outcome_name+'(data,**kwargs_fun)')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
