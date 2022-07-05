import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split as datasplit
from clustering_algos import agglomerative_clustering, kmeans_clustering, box_clustering
from sklearn.ensemble import GradientBoostingRegressor as GDBR


##### Evaluations ATT
def computeATT_per_cluster(data, treatment = "Treatment", cluster_name = None,
                               outcome = 'Y', counterfactual = 'Ycf', ITE_name = 'ITE', 
                               ATT_CLUST_name = 'ATT_CLUST', points_name = '#CLUST_POINTS', weight_names = '#CLUST_WEIGHTS',
                          hom_name = 'CLUST_HOMOGENEITY', class_name = 'C', att = True):
    """
    
    Computes the ATT for each cluster
    with a known Counter Factual Outcome
    counterfactual(string): column name of the counterfactual
    
    """
    if att:
        filt = data[treatment] == 1
        data2 = data[filt].reset_index(drop = True)
    else:
        data2 = data
    #data2[ITE_name] =  data2[outcome]-data2[counterfactual]
    
    res = data2.groupby(cluster_name, sort = True).apply(lambda x: pd.Series(
                                                    [x[ITE_name].mean(), len(x),
                                                    x[class_name].value_counts().max()/x[class_name].value_counts().sum()],
                                                     index = [ATT_CLUST_name, points_name, hom_name])).reset_index()
    res[weight_names] = res[points_name].values/res[points_name].sum()
    return res
    
def computeITE(data, treatment = "Treatment",
                               outcome = 'Y', counterfactual = 'Ycf', ITE_name = 'ITE'):
    
    filt = data[treatment] == 1
    data2 = data[filt].reset_index(drop = True)
    data2[ITE_name] =  data2[outcome]-data2[counterfactual]
    
    return data2




def predict_cf(data, col_select = None, f1 = None, f2 = None,
               f1_kwargs = None, f2_kwargs = None, treatment_name = 'Treatment',
              cf_name = 'Ycf-model', outcome = 'Y', ite_name = 'ITE-model'):
    
    """
    predict counterfactual and ITE based
    on two approaches:
    1. E[Y|X,T]
    2. E[Y|X,T=1], E[Y|X, T=0]
    
    data: DataFrame
    col_select: columns for learning
    f1: class of models for first model(if the  approach 2. is used)
        or class of models to calculate 1.
    f2: class of models for second model(if approach 2. is followed else None)
    f1_kwargs:dict with fiiting params
    f2_kwargs: same as above
    treatment_name:(string) name of treatment
    cf_name:(string) name to be given for the predicted counterfactual
    outcome:(string) name for the outcome (our target)
    ite_name:(string) name for the calculated ITE
    
    returns the updated data with the counterfactual and the ite
    """
    
    tr, ts = datasplit(data, test_size = 0.2, random_state = 0)
    data['IS_TEST'] = 0
    data.loc[ts.index, 'IS_TEST'] = 1
    data[cf_name] = 0
    data[ite_name] = 0
    if not f2:
        
        model = f1(**f1_kwargs)
        model = model.fit(tr[col_select].values)
        data[cf_name] = model.predict(data[col_select].values, data[outcome].values)
        return data, [model]
    
    else:
        
        filt = tr[treatment_name] == 1
        model1 = f1(**f1_kwargs)
        model1 = model1.fit(tr.loc[filt, col_select].values, tr.loc[filt, outcome].values)
        model2 = f2(**f2_kwargs)
        model2 = model2.fit(tr.loc[~filt, col_select].values, tr.loc[~filt, outcome].values)
        
        filt2 = data[treatment_name] == 1
        data.loc[filt2, cf_name] = model2.predict(data.loc[filt2, col_select].values)
        data.loc[~filt2, cf_name] = model1.predict(data.loc[~filt2, col_select].values)
        data.loc[filt2, ite_name] = data.loc[filt2,outcome]-data.loc[filt2,cf_name]
        data.loc[~filt2, ite_name] = -data.loc[~filt2,outcome]+data.loc[~filt2,cf_name]

    
        return data, [model1, model2]
        
        
    
def calculate_ite(data, treatment = 'Treatment', counterfactual = 'Ycf', outcome = 'Y', ite_name = 'ITE'):
    """
    data: pandas array with relevant data
    treatment:(string) name of the treatment columns
    counterfactual:(string) name of the counterfactual outcome column
    outcome:(string) name of the outcome column
    ite_name:(string) name of the new column that will be created containing the ite
    
    returns data with the extra ite column based on counterfactual and factual outcomes
    
    
    """
    filt = data[treatment] == 1
    data[ite_name] = 0
    
    data.loc[filt, ite_name] = data.loc[filt, outcome] - data.loc[filt, counterfactual]
    data.loc[~filt, ite_name] = -data.loc[~filt, outcome] + data.loc[~filt, counterfactual]
    
    return data

def get_homogeneity(data, clusters, col_select = None, clustering_algo = None, fit_kwargs = {}, **kwargs):
    """
    Homogeneity experiment
    data: generated data
    clusters:(list) clusters to examine
    clustering_algo:[list] names of clustering algorithms to examine
    fit_kwargs: fit parameters for clustering algorithms
    kwargs: other key-word parameters for clustering algorithms
    
    """
    result = {}
    result['clusters'] = clusters
    for algo in clustering_algo:
        print('Running:'+algo)
        exp = []
        for cluster in clusters:
            print('Number of Clusters:'+str(cluster)+' algo'+algo)
            data2,_ = eval(algo+'(data.copy(), clusters=cluster,'+ 
                           'col_select = col_select, cluster_name="A",'+
                           'fit_kwargs=fit_kwargs, **kwargs)')
            
            res = computeATT_per_cluster(data2.copy(), hom_name = 'HOM',weight_names = 'W', cluster_name  = "A")
            hom = (res['HOM']*res['W']).sum(axis = 0)
            exp.append(hom)
            
        result[algo] = exp
    
    return pd.DataFrame(result)

def generate_paths(K = 10**4, low = 0, high = 1, Cdraw = None, Cdraw_name = 'Box-Cl2'):
    
    """
    generate the coordinates of the boxes as well
    as the colors based on the clustering
    
    """
    
    box_per_dim = int(K**(1/2))
    bl = (high-low)/box_per_dim
    
    boxes_vertices = []
    colors = []
    counter = 0
    for i in range(box_per_dim):
        for j in range(box_per_dim):
            vertices = [[i*bl, j*bl],
                       [i*bl+bl, j*bl],
                       [i*bl, j*bl+bl],
                        [i*bl+bl, j*bl+bl]]
            boxes_vertices.append(vertices)
            colors.append(Cdraw[Cdraw_name].iloc[counter])
            counter += 1
            
    return boxes_vertices, colors

def confusion_matrix(data, true_col, predicted_col, unique_values, save = False, save_dir = None):
    ctrue = data[true_col].values
    cpred = data[predicted_col].values
    
    conf_mat = np.zeros([unique_values, unique_values])
    
    for i in range(unique_values):
        toti = np.sum(ctrue == i)
        for j in range(unique_values):
            cij = np.sum((ctrue == i) & (cpred == j))
            conf_mat[i,j] = cij/toti
            
    cols = ['C{} Pred'.format(i) for i in range(unique_values)]
    indx = ['C{} True'.format(i) for i in range(unique_values)]
    
    mat = pd.DataFrame(conf_mat, index = indx, columns = cols)
    
    if save_dir:
        name = save_dir + predicted_col +'.csv'
        mat.to_csv(name)
    
    return mat
           
                

