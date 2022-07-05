import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from sklearn.cluster import KMeans 
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from numba import jit
import numba as nb

#1. box clustering
def box_clustering(data, clusters = 100,cluster_name ='Box-Cluster',
                   min_support=0, max_support=1, col_select = None, fit_kwargs = {}):
    """
    data: pandas array with data to apply box clustering
    clusters:(number of boxes) clusters
    min_support:(float) the minimum value the box clustering is considering
    max_support:(float) the maximum value the box clustering is considering
    box_length(deprecated):(float) the length of each box
    col_select:[list of strings] names of features to consider for clustering
    
    returns the dataframe with the clustering applied
    """
    values = data[col_select].values
    boxes_per_dim = int(clusters**(1/values.shape[1]))
    box_length = (max_support-min_support)/boxes_per_dim
    grid_length = int((max_support-min_support)/box_length)
    coord_float = np.floor((values-min_support)/box_length).astype('int')
    pivot_array = np.array([grid_length**i for i in range(values.shape[1])])
    index_array = np.sum(coord_float*pivot_array, axis = 1).astype(int)
    
    #values_pd = pd.DataFrame(values, columns = x_names)
    data[cluster_name] = index_array
    
    return data, boxes_per_dim

#2. Kmeans
def kmeans_clustering(data, clusters=100, cluster_name = 'Kmeans-Cluster', 
                      col_select=None,fit_kwargs = {}, assign_means = False, means = None,  **kwargs):
    """
    
    data: pandas array with data to apply kmeans clustering
    clusters:(number of boxes) clusters
    
    """
    if len(col_select) == 1:
        values = data[col_select].values.reshape(-1,1)
    else:
        values = data[col_select].values
        
    kmeans = KMeans(n_clusters = clusters, **kwargs)
    if assign_means:
        kmeans = KMeans(n_clusters = clusters,max_iter=1)
        kmeans = kmeans.fit(values)
        kmeans.cluster_centers_ = means
    else:
        kmeans = kmeans.fit(values,**fit_kwargs)
    
    if len(col_select) == 1:
        #if clusters == 2:
        centers = np.sort(kmeans.cluster_centers_, axis = 0)
        #c0 = kmeans.cluster_centers_[0].copy()
        #c1 = kmeans.cluster_centers_[1].copy()
        kmeans.cluster_centers_ = centers
        #if c0 >= c1:
        #    kmeans.cluster_centers_[0] = c1
        #    kmeans.cluster_centers_[1] = c0
                
            
    data[cluster_name] = kmeans.predict(values)
    
    return data, kmeans
    
#3. Spectral
def spectral_clustering(data, clusters=100, cluster_name = 'Spectral-Cluster', col_select=None,fit_kwargs = {}, **kwargs):
    """
    
    data: pandas array with data to apply box clustering
    clusters:(number of boxes) clusters
    
    """
    if len(col_select) == 1:
        values = data[col_select].values.reshape(-1,1)
    else:
        values = data[col_select].values
        
    spec = SpectralClustering(n_clusters = clusters, **kwargs)
    spec = spec.fit(values,**fit_kwargs)
    data[cluster_name] = spec.labels_
    
    return data, spec

#4. Agglomerative
def agglomerative_clustering(data, clusters=100, cluster_name = 'Agglomerative-Cluster', col_select=None,fit_kwargs = {}, **kwargs):
    """
    
    data: pandas array with data to apply box clustering
    clusters:(number of boxes) clusters
    
    """
    if len(col_select) == 1:
        values = data[col_select].values.reshape(-1,1)
    else:
        values = data[col_select].values
        
    agglo = AgglomerativeClustering(n_clusters = clusters, **kwargs)
    agglo = agglo.fit(values,**fit_kwargs)
    data[cluster_name] = agglo.labels_
    
    return data, agglo

def optimal_clustering(data, max_clusters=7, threshold = 0.5,  cluster_name = 'Optimal-Cluster',
                       col_select=None, fit_kwargs = {}, N = 100, val = 3,  **kwargs):
    """
    1-D optimal clustering
    data: pandas arrays with column to perform the optimal 1 d clustering
    max_clusters: maximum number of clusters optimal clustering to consider
    threshold: where to stop optimal clustering (preseted for our problem)
    cluster_name: how to name the new column with clustering
    col_select: col name to perform clustering
    fit_kwargs: not used just for framework consistency
    N: number of datapoints for pre setted threshold
    **kwargs: not used just for framewrok consistency
    """
    
    #presetted threshold for our problem
    threshold = 3.8*np.log(N)/np.sqrt(N)
    
    #sort data by the column to cluster 
    data2 = data.copy().sort_values(by= col_select)
    values = data2[col_select].values
    index = np.array(data2.index)
    
    #perform optimal kmeans
    D,B = opt_kmeans(values, max_clusters)
    D = D/np.sqrt(N)
    
    #find the optimal cluster number
    opt_clust = 0
    for k in range(max_clusters):
        if D[-1,k] <= threshold:
            opt_clust = k+1
            break
    if opt_clust != val:
        print('Did not pick '+str(val)+', it picked:', opt_clust)
        opt_clust = val
    #find the intervals of the clusters
    intervals = backtrack(B, len(values), opt_clust)
    
    #calculate the means and the indexes correspodning in each cluster
    means = []
    indx_class = {}
    class_counter = 0
    for i in range(1,len(intervals)):
        mu = np.mean(values[int(intervals[i-1]):int(intervals[i])])
        indxs = index[int(intervals[i-1]):int(intervals[i])]
        indx_class[class_counter] = indxs
        class_counter += 1
        means.append(mu)
        
    #assign cluster indexes to datapoints
    data[cluster_name] = 0
    data2[cluster_name] = 0
    for key, val in indx_class.items():
        data.loc[val, cluster_name] = key
        data2.loc[val, cluster_name] = key
    #reset data index
    data2 = data2.reset_index(drop = True)  
    
    return data, np.array(means)

######## OPTIMAL CLUSTERING

"""Deprecated OPT CL
def backtrack(B, n, k):
    intervals = [n]
    nex = n+1
    for j in range(k, 1,-1):
        intervals.append(int(B[nex-1, j]))
        nex = intervals[-1]
        
    intervals.append(0)
    return intervals[::-1]

def opt_kmeans(data, k):
    D = np.zeros([len(data)+1,k+1])
    B = np.zeros([len(data)+1,k+1])
    
    dij = 0
    mij = 0
    for i in range(1,len(D)):
        D[i,0] = float('inf')
        dij = dij+ (i-1)/i*(data[i-1]-mij)**2
        D[i,1] = dij
        mij = (data[i-1]+(i-1)*mij)/i
        
    for k in range(2,k+1):
        for i in range(k, len(data)+1):
            temp = float('inf')
            tmp_indx = -1
            dji = 0
            mji = data[i-1]
            for j in range(i,k-1,-1):
                #print(j)

                if i == j:
                    dji = 0
                    mji = data[i-1]
                else:
                    dji = dji + (i-j)/(i-j+1)*(data[j-1]-mji)**2
                    mji = (data[j-1]+(i-j)*mji)/(i-j+1)
                #if i <=3:
                    #print(i,j,dji,mji)
                if dji + D[j-1,k-1] < temp:
                    temp_indx = j
                temp = min(temp, dji + D[j-1,k-1])
            D[i,k] = temp
            B[i,k] = temp_indx
            
    return D,B

def opt_kmeans_with_weights(data, k, weights = None):
    if not weights.size:
        weights = np.ones(len(data))
    
    D = np.zeros([len(data)+1,k+1])
    B = np.zeros([len(data)+1,k+1])
    weights = np.ones(shape = [len(data)])
    #weights[0:100] = 1000000
    #weights = weights/np.sum(weights)  
    dij = 0
    mij = 0
    wc = 0
    #initialize
    for i in range(1,len(D)):
        D[i,0] = float('inf')
        dij = dij+ wc/(wc+weights[i-1])*(data[i-1]-mij)**2
        D[i,1] = dij
        mij = (data[i-1]*weights[i-1]+wc*mij)/(wc+weights[i-1])
        wc = wc+ weights[i-1]
        
    for k in range(2,k+1):
        for i in range(k, len(data)+1):
            temp = float('inf')
            tmp_indx = -1
            dji = 0
            mji = 0
            wcji = 0
            for j in range(i,k-1,-1):
                #print(j)

                if i == j:
                    dji = 0
                    mji = data[i-1]
                    wcji = weights[i-1]
                else:
                    dji = dji + wcji/(wcji+weights[j-1])*(data[j-1]-mji)**2
                    mji = (data[j-1]*weights[j-1]+wcji*mji)/(wcji+weights[j-1])
                    wcji +=weights[j-1]
                #if i <=3:
                    #print(i,j,dji,mji)
                if dji + D[j-1,k-1] < temp:
                    temp_indx = j
                temp = min(temp, dji + D[j-1,k-1])
            D[i,k] = temp
            B[i,k] = temp_indx
       
    return D,B

"""
@jit(nopython = True)
def opt_kmeans(data, k):
    D = np.zeros((data.shape[0], k ), dtype = np.float64)
    B = np.zeros((data.shape[0], k ), dtype = np.int64)
    data = data.astype(np.float64)
    
    di1 = 0
    mi1 = data[0]
    for i in range(1, D.shape[0]):
        di1 = di1 + (i)/(i+1)*(data[i]-mi1)**2
        D[i,0] = di1
        mi1 = (data[i]+(i)*mi1)/(i+1)
        
    for k2 in range(1,k):
        for i in range(k2, data.shape[0]):
            temp = np.inf
            tmp_indx = -1
            dji = 0
            mji = data[i]
            for j in range(i,k2-1,-1):
                #print(j)

                if i == j:
                    dji = 0
                    mji = data[i]
                else:
                    dji = dji + (i-j)/(i-j+1)*(data[j]-mji)**2
                    mji = (data[j]+(i-j)*mji)/(i-j+1)
                #if i <=3:
                #print(i,j,dji + D[j-1,k2-1],temp, mji)
                if dji + D[j-1,k2-1] < temp:
                    temp_indx = j
                temp = np.minimum(temp, dji + D[j-1,k2-1])
            #print(temp, j, k2)
            D[i,k2] = temp
            B[i,k2] = temp_indx
            
    return D,B

def backtrack(B, n, k):
    intervals = [n]
    nex = n
    for j in range(k-1, 0,-1):
        intervals.append(int(B[nex-1, j]))
        nex = intervals[-1]
        
    intervals.append(0)
    return intervals[::-1]




#ADAPTIVE BOX
def adaptive_box(data, clusters = 100, cluster_name ='Box-Cluster',ite_old_name = 'ITE', ite_name = 'Adaptive-ite',
                   min_support=0, max_support=1, col_select = None, fit_kwargs = {}):
    
    """
    data: pd.dataFrame with one column containing the box-clustering (box# the datapoint belongs to)
    clusters: number of total boxes in the clustering
    cluster_name: the name of the box clustering column in data
    ite_old_name: name of the current calculated ite for a point
    ite_name: the name the new adaptive ite will take
    min_support: support of the data
    max_support the max support of the data
    col_select: the columns containing the data points
    fit_kwargs: Nothing just for framework consistency
    """
    
    #add the new column to data
    data[ite_name] = 0
    
    #grouped by cluster data
    data_groups = data.set_index(cluster_name).copy()
    
    #take datapoints
    values = data[col_select].values
    dims = values.shape[1]
    #boxes_per_dimension
    grid_length = int(clusters**(1/dims))
    
    #espilon side of the box
    box_length = (max_support-min_support)/grid_length
    
    #array to put (x,y) pair to the correct box number
    pivot_array = np.array([grid_length**i for i in range(dims)])
    
    offset = [-box_length, 0, box_length]
    offset_np = np.zeros([9,2])
    cn = 0
    for offs_x in offset:
        for offs_y in offset:
            offset_np[cn] = np.array([offs_x,offs_y])
            cn += 1
                
    #for each point in data
    for  i, row in enumerate(values):
        #if i%50 == 0:
           # print(i)
        box_numbers = set()
        for offs in offset_np:
            xnew = row + offs
            #support check
            supp_check = all((xnew >= min_support) & (xnew <= max_support))
            if not supp_check:
                continue
            #get box number   
            xnew_grid_coord = np.floor((xnew-min_support)/box_length).astype(int)
            #print(xnew_grid_coord.shape, pivot_array.shape)

            bn = np.sum(xnew_grid_coord*pivot_array).astype(int)

            box_numbers.add(bn)
        
       # print(box_numbers, i)
        candidates = data_groups.loc[box_numbers,col_select+[ite_old_name ]].values
        count = 0
        avg = 0

        for  cand in candidates:
            vals = np.abs(row-cand[0:dims])
            if all(vals <= box_length):
                avg += cand[-1]
                count+=1
        
        new_ite = avg/count
        data.loc[i, ite_name] = new_ite
        
    return data





############################ FASTER ADAPTIVE
_spec = [
    ('data_groups2', nb.types.List(nb.types.Array(nb.types.float64, 3, 'C')))
    
]
def adaptive_box2(data, clusters = 100, cluster_name ='Box-Cluster',ite_old_name = 'ITE', ite_name = 'Adaptive-ite',
                   min_support=0, max_support=1, col_select = None, fit_kwargs = {}):
    
    """
    data: pd.dataFrame with one column containing the box-clustering (box# the datapoint belongs to)
    clusters: number of total boxes in the clustering
    cluster_name: the name of the box clustering column in data
    ite_old_name: name of the current calculated ite for a point
    ite_name: the name the new adaptive ite will take
    min_support: support of the data
    max_support the max support of the data
    col_select: the columns containing the data points
    fit_kwargs: Nothing just for framework consistency
    """
    
    #add the new column to data
    data[ite_name] = 0
    
    #grouped by cluster data
   # data_groups = data.set_index(cluster_name).copy()
    
    # groups dictionary
    names = col_select + [ite_old_name]
    data_groups2 = data.groupby(cluster_name, 
                                 sort = True).apply(lambda x: x[['x0','x1','ITE']].values.astype(np.float64)).values
    
    data_groups2 = list(data_groups2)
    #new ites
    new_ites = np.zeros(len(data)).astype(np.float64)
    
    #take datapoints
    values = data[col_select].values.astype(np.float64)
    
    #feature length
    f = values.shape[1]
    
    
    
    dims = values.shape[1]
    #boxes_per_dimension
    grid_length = int(clusters**(1/dims))
    
    #espilon side of the box
    box_length = (max_support-min_support)/grid_length
    
    #array to put (x,y) pair to the correct box number
    pivot_array = np.array([grid_length**i for i in range(dims)]).astype(np.float64)
    
    offset = [-box_length, 0, box_length]
    offset_np = np.zeros([9,2])
    cn = 0
    for offs_x in offset:
        for offs_y in offset:
            offset_np[cn] = np.array([offs_x,offs_y])
            cn += 1
    offset_np = offset_np.astype(np.float64)           
    #for each point in data
    #print(type(data_groups2), data_groups2[0].dtype, new_ites.dtype, values.dtype, pivot_array.dtype,
         # offset_np.dtype)
    new_ites = adapt_numba(data_groups2,new_ites, values, pivot_array,
                           offset_np, max_support, 
                           min_support, box_length, dims)
    
    data[ite_name]  = new_ites  
    return data

@jit(nopython = True)
def adapt_numba(data_groups2, new_ites, values, pivot_array,
                offset_np, max_support, min_support, box_length, dims):
    for  i in range(values.shape[0]):
        #if i%50 == 0:
           # print(i)
        box_numbers = set()
        for k in range(offset_np.shape[0]):
            xnew = values[i] + offset_np[k]
            #support check
            supp_check = np.all((xnew >= min_support) & (xnew <= max_support))
            if not supp_check:
                continue
            #get box number   
            xnew_grid_coord = np.floor((xnew-min_support)/box_length).astype(np.int64)
            #print(xnew_grid_coord.shape, pivot_array.shape)

            bn = np.sum(xnew_grid_coord*pivot_array)#.astype(np.int64)
            bn = int(bn)
            box_numbers.add(bn)
       # print(box_numbers, i)
        #candidates = data_groups.loc[box_numbers,col_select+[ite_old_name ]].values
        count = 0
        avg = 0
        box_numbers = list(box_numbers)

        for bx in box_numbers:
            for  j in range(data_groups2[bx].shape[0]):
                cand = data_groups2[bx][j]
                vals = np.abs(values[i]-cand[0:dims])
                if np.all(vals <= box_length):
                    avg += cand[-1]
                    count+=1
        
        new_ite = avg/count
        new_ites[i] = new_ite
        
    return new_ites






















