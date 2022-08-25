# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 10:05:23 2022

@author: Mahdi Orooji
"""

import numpy as np
import _pickle as pk
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from scipy.io import savemat

def Clusters_division(Times_array,Clustering_result,n_clusters):
    Clutsers = dict()
    for x in range(0,n_clusters):
        Clutsers[x] = Times_array[Clustering_result==x]
    return Clutsers

def plot_clusters(Clusters,n_clusters, val):
    plt.figure(figsize=(10,5), dpi=150)
    Legends = ["Cluster " + str(x) for x in range(0,n_clusters)]
    for x,y in Clusters.items():
        plt.plot(y, np.zeros_like(y) + val, 'x')
    plt.ylim(-0.01, 0.01)
    plt.legend(Legends)
    plt.title("Scatter plot and clustering result with K="+str(n_clusters))
    plt.xlabel("SAT Runtime")
    plt.savefig("K-means_result_n="+ str(n_clusters) + ".jpg")
    plt.show()
    plt.close()
    
    
CircuitNames = ['74181','74182','74283','74L85','c1196','c1238','c1355','c17',
 'c2670','c3540','c432','c499','c5315','c6288','c7552','c880','s1196','s1196a',
 's1238','s1238a']

All_times = []
for each_benchmark in CircuitNames:
    Times = pk.load(open(os.path.join(os.getcwd(),each_benchmark + "_Y.pk"), 'rb'))
    All_times.extend(Times)
      
Times_array = np.asarray(All_times)
Times_array = np.expand_dims(Times_array,axis=1)

### Removing -1 values from the Times_array
r,c = np.where(Times_array == -1)
Times_array = np.delete(Times_array, r)
Times_array = np.expand_dims(Times_array,axis=1)

Number_of_clusters = 5
Each_clutsering_result = dict() 
Criterions = dict() 
for n_clusters in range(2,Number_of_clusters+1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(Times_array)  
    Clustering_result = kmeans.labels_
    Criterions[n_clusters] = kmeans.inertia_
    Clusters = Clusters_division(Times_array,Clustering_result,n_clusters)
    Each_clutsering_result[n_clusters] = Clusters
    plot_clusters(Clusters,n_clusters, 0)