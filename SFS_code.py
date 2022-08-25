# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:05:12 2022

@author: Mahdi Orooji
"""

import numpy as np
#import pickle as pk
#import os
import sklearn
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt 
from scipy.io import savemat

from util import read_ic

def Best_Comb(Results):
    Avg_metrics = []
    for index, each_result in Results.items():
        Avg_metrics.append(each_result["avg_score"])
    Best_com_index = Avg_metrics.index(max(Avg_metrics))
    return Results[Best_com_index+1]['feature_idx'], np.asarray(Avg_metrics) 

CircuitNames = ['c1355','c2670','c3540','c5315','c6288','c7552']
CircuitNames = [CircuitNames[1]]

Number_of_scaler_features = 40 
m2 = 59830
_, X, Y, _, _, _ = read_ic(CircuitNames,Number_of_scaler_features,m2,False)
    
#### Loading Dataset     
#X = pk.load(open(os.path.join(os.getcwd(),"c499_X.pk"), 'rb'))
#Y = pk.load(open(os.path.join(os.getcwd(),"c499_Y.pk"), 'rb'))

X = np.asarray(X)
Y = np.asarray(Y)

### Feature Selection 
from sklearn.neighbors import KNeighborsClassifier
Model = KNeighborsClassifier()
sfs1 = SFS(Model,
           k_features = 40, 
           forward = True,
           cv = 10)

sfs1.fit(X,Y)
Results = sfs1.subsets_
SF, Metrics = Best_Comb(Results)

#SF = Results[3]['feature_idx']
fig = plot_sfs(sfs1.get_metric_dict())
plt.savefig('my_plot.png')
plt.close

Mydic = {"Selected_Features":SF, "Metrics":Metrics}
savemat('SF.mat',Mydic)

