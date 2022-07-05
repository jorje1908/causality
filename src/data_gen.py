from sklearn.datasets import make_blobs, make_classification
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel 

class MakeData(object):
    """
    A class to generate
    Datasets
    
    """
    
    def __init__(self):
        
        return 
    
    
    def make_blobs(self, n_samples=100,
                            n_features=2, 
                            centers=None, 
                            cluster_std=1.0, 
                            center_box= (-10.0, 10.0),
                            shuffle=True, 
                            random_state=None, 
                            return_centers= True,
                            prior_centers  = None, 
                            mixture_number = 2, 
                            mixture_proportions = None, 
                            samples_per_mixture = None):
        
        
        """
        wrapper for make blobs in scikit learn
        see info there:
        'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.
        html#sklearn.datasets.make_blobs'
        prior_centers: array-like(n_centers, n_features)
                        generate gaussian mixtures with means
                        sampled from these prior centers
                        
        mixture_number: int, default 2
                        how many mixtures to generate
                        from each prior center
                        
        mixture_proportions" array-like(n_centers, mixture_number)
                            propabilities a point to be generated
                            from a gaussian from a specific mixture
                            the sum of each row should add up to 1
        samples_per_mixture: int, or array-like(mixture_number)
        
        
        """ 
        if prior_centers is not None:
            centers, n_samples = self.generate_mixtures(prior_centers,
                                                        mixture_number, mixture_proportions, 
                                                       samples_per_mixture,
                                                       n_features)
        
        parameters = {'n_samples':n_samples,
                      'n_features':n_features,
                      'centers': centers,
                      'cluster_std':cluster_std, 
                      'center_box':center_box,
                      'shuffle':shuffle, 
                      'random_state':random_state, 
                      'return_centers':return_centers}
        
        if return_centers:
            X,Y, centers = make_blobs(**parameters)
            return X, Y, centers
        else:
            X,Y = make_blobs(**parameters)
            return X,Y
            
            
    def generate_mixtures(self, prior_centers, mixture_number, 
                          mixture_proportions, samples_per_mixture, n_features):
        
        prior_centers = np.repeat(prior_centers, n_features, axis = 1)
        centers = np.zeros(shape = [len(prior_centers)*mixture_number, n_features])
        n_samples = np.zeros(len(centers))
        
        if mixture_proportions is None:
            mixture_proportions = np.ones(shape = [len(prior_centers), mixture_number])/mixture_number
            
        cov = np.eye(n_features)
            
        for i, center in enumerate(prior_centers):
            for j in range(mixture_number):
                centers[i*mixture_number + j] = np.random.multivariate_normal(center, cov)
                                    
                                    
        for i, s in enumerate(samples_per_mixture):
            ss = np.random.multinomial(s, pvals = mixture_proportions[i,:])
            for j in range(mixture_number):
                n_samples[i*mixture_number+j] = ss[j]
                
        return centers, n_samples.astype(int)
            
        
        
    def plot_data(self, X, Y, names = None):
        
        if X.shape[1] > 2:
            pca = PCA(n_components = 2)
            X = pca.fit_transform(X)
            
        clusters = np.unique(Y)  
        fig, ax = plt.subplots(1,1)
        
        for i in range((len(clusters))):
            
            xi = X[Y==i]
            ax.scatter(xi[:,0], xi[:,1], s = 0.1, label = '' if  names is None else names[i])
            if not names is None:
                ax.legend(loc = 'best',markerscale=10)
            
        
       # plt.show()
        return fig, ax
    
    
    def assign_treatement(self, Y = None,  treated_size = 500, eligible_clusters = None,
                        eligible_proba = 0.99, non_eligible_proba = 0.05):
        
        """
        Y = nx1 array denoting the cluster numbers for each data point
        treated_size = Subjects in the study to be treated
        eligible_clusters = clusters of people eligible
        eligible_proba: P(T = 1 | E = 1)
        non_eligible_proba: 
        
        """
        
        
        if (Y is None) or (eligible_clusters is None):
            print('Give cluster ids and eligible clusters')
            
        eligible = np.zeros_like(Y)
        
        #denote eligible people with 1s
        mask = np.isin(Y, eligible_clusters)
        eligible[mask] = 1
        
        #Treatement assignments
        uniform_dist = np.random.uniform(0,1, len(Y))
        uniform_dist[mask] = uniform_dist[mask] <= eligible_proba
        uniform_dist[~mask] = uniform_dist[~mask] <= non_eligible_proba
        
        
        
        #Sample For treatment
        treatment_indexes = np.random.choice(np.where(uniform_dist == 1)[0], treated_size, replace = False)
        treatment = np.zeros_like(uniform_dist)
        treatment[treatment_indexes] = 1
        
        eligible = eligible.reshape([-1,1])
        treatment = treatment.reshape([-1,1])
        Y = Y.reshape([-1,1])
        
        df =  pd.DataFrame(np.concatenate([eligible, treatment, Y], axis = 1), columns = ['E', 'Tr', 'C'])
        df.Tr = df.Tr.astype(bool)
        return df
    
    def assign_constant_outcome(self, X, out_el_t = 1, out_el_nt = 0, out_nel_t = 1, out_nel_nt = 1, 
                               flip = 0.05):
        """
        X[E,T,C]: matrix with treatment eligibility, clusters
        
        """
        X['Y'] = 0
        
        X.loc[(X.Tr == 1)&(X.E == 1), 'Y'] = out_el_t
        X.loc[(X.Tr == 0)&(X.E == 1), 'Y'] = out_el_nt
        X.loc[(X.Tr == 1)&(X.E == 0), 'Y'] = out_nel_t
        X.loc[(X.Tr == 0)&(X.E == 0), 'Y'] = out_nel_nt
        X['Y1'] = 0
        X['Y0'] = 0
        X.loc[(X.Tr == 1)&(X.E == 1), 'Y1'] = out_el_t
        X.loc[(X.Tr == 0)&(X.E == 1), 'Y1'] = out_el_t
        X.loc[(X.Tr == 1)&(X.E == 0), 'Y1'] = out_nel_t
        X.loc[(X.Tr == 0)&(X.E == 0), 'Y1'] = out_nel_t
        
        X.loc[(X.Tr == 1)&(X.E == 1), 'Y0'] = out_el_nt
        X.loc[(X.Tr == 0)&(X.E == 1), 'Y0'] = out_el_nt
        X.loc[(X.Tr == 1)&(X.E == 0), 'Y0'] = out_nel_nt
        X.loc[(X.Tr == 0)&(X.E == 0), 'Y0'] = out_nel_nt
        
        #random flip of outcome
        uni = np.random.uniform(size = len(X)) <= flip
        
        X.loc[uni,'Y'] = 1-X.loc[uni,'Y']
        X.loc[X.Tr == 1, 'Y1'] = X.loc[X.Tr == 1, 'Y']
        X.loc[X.Tr == 0, 'Y0'] = X.loc[X.Tr == 0, 'Y']

        
        return X
    
    
    def printstats(self, X):
        print('Total Samples:', len(X))
        print('Total Eligible', (X.E == 1).sum())
        print('Total Non Eligible', (X.E == 0).sum())
        print('Total Eligible in Treatement', ((X.E == 1) & (X.Tr == 1)).sum())
        print('Total Non Eligible in Treatement', ((X.E == 0) & (X.Tr == 1)).sum())
        print('Total Eligible in Non Treatement', ((X.E == 1) & (X.Tr == 0)).sum())
        print('Total Non Eligible in Non Treatement', ((X.E == 0) & (X.Tr == 0)).sum())

              
        return
    
    @staticmethod
    def calculateEffects(X, out = True):
        ATE = (X.Y1 - X.Y0).mean()
        
        mask = X.Tr == 1
        ATT = (X[mask].Y1-X[mask].Y0).mean()
        
        maskE =  X.E == 1
        ATE_eligible = (X[maskE].Y1-X[maskE].Y0).mean()
        
        maskET =  (X.E == 1)& (X.Tr == 1)
        ATT_eligible = (X[maskET].Y1-X[maskET].Y0).mean()
        
        maskNE =  X.E == 0
        ATE_Neligible = (X[maskNE].Y1-X[maskNE].Y0).mean()
        
        maskNET =  (X.E == 0)& (X.Tr == 1)
        ATT_Neligible = (X[maskNET].Y1-X[maskNET].Y0).mean()
        
        if out:
            print('Real ATE:',ATE)
            print('Real ATT:',ATT)
            print('Real ATE Eligible:',ATE_eligible)
            print('Real ATT Eligible:',ATT_eligible)
            print('Real ATE Non Eligible:',ATE_Neligible)
            print('Real ATT Non Eligible:',ATT_Neligible)

        names = ['ate', 'att', 'ateE', 'attE', 'ateNE', 'attNE']
        metrics = [ATE, ATT, ATE_eligible, ATT_eligible, ATE_Neligible,
                  ATT_Neligible]
        
        true_effects = pd.Series(metrics, index = names)
        
        return true_effects




    
    
    def combine(self, X, D):
        
        X = pd.DataFrame(X, columns = ['x'+str(i) for i in range(X.shape[1])])
        
        F = pd.concat((X, D), axis = 1, ignore_index = False)
        return F



from sklearn.base import BaseEstimator, ClassifierMixin

class MLWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    
        
        
        
def log_effect(model, data, causes_names, w = None):
    
    m0 = model(random_state = 0, class_weight = w[0])
    mask0 = data.Tr == 0
    m0 = m0.fit(X = data.loc[mask0, causes_names].values,
                y = data[mask0].Y.values)
    
    mask1 = data.Tr == 1
    m1 = model(random_state = 0, class_weight = w[1])
    m1 = m1.fit(X = data.loc[mask1,causes_names].values,
                y = data[mask1].Y.values)
    
    return m0, m1

def ate(Y, Yest, Tr ):
    
    mask1 = Tr == 1
    ate = np.sum(Y[mask1] - Yest[mask1].Y0est)
    mask0 = Tr == 0
    ate = ate + np.sum(Yest[mask0].Y1est - Y[mask0])
    
    return ate/len(Y)

def att(Y, Yest, Tr):
    
    mask1 = Tr == 1
    K  = np.sum(mask1)
    att = np.sum(Y[mask1] - Yest[mask1].Y0est)
    
    return att/K
       
        
def causalModel(data = None, treatment_name = None, common_causes_names = None,
               outcome_name = None):
    
    data = data.copy()
    model = CausalModel(data = data, treatment = treatment_name,
                   common_causes = common_causes_names, outcome = outcome_name,
                   proceed_when_unidentifiable = True)
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    return model, identified_estimand , data      
        