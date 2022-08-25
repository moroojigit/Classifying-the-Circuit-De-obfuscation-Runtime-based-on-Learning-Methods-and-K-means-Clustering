# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:30:56 2022

@author: Mahdi Orooji
"""

import numpy as np
import _pickle as pk
import os
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from scipy.io import savemat
from yellowbrick.cluster import KElbowVisualizer
    
    
CircuitNames = ['74181','74182','74283','74L85','c1196','c1238','c1355','c17',
 'c2670','c3540','c432','c499','c5315','c6288','c7552','c880','s1196','s1196a',
 's1238','s1238a']

All_times = []
for each_benchmark in CircuitNames:
    Times = pk.load(open(os.path.join(os.getcwd(),each_benchmark + "_Y.pk"), 'rb'))
    All_times.extend(Times)
      
Times_array = np.asarray(All_times)
Times_array = np.expand_dims(Times_array,axis=1)

"""
Removing -1 values from the Times_array
r,c = np.where(Times_array == -1)
Times_array = np.delete(Times_array, r)
Times_array = np.expand_dims(Times_array,axis=1)
"""
kmeans = KMeans()
Max_Number_clusters = 7

elbow_obj = KElbowVisualizer(kmeans, k=Max_Number_clusters,
                             metric='silhouette', 
                             timings=False)
# Fit the data to the elbow technique for determining the optimum number of clusters
elbow_obj.fit(Times_array) 
elbow_obj.show(outpath="elbow_result.png") 