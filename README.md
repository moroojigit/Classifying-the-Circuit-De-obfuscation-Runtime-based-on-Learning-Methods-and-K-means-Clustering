# Classifying-the-Circuit-De-obfuscation-Runtime-based-on-Learning-Methods-and-K-means-Clustering

## Introduction
<p align="justify"> In this work, we proposed an automatic framework based on deep learning methods to recognize the resistance or vulnerability of ICs. we first assigned labels to each IC based on the runtime of SAT attacks. To this end, elbow technique was employed to determine the best number of clusters of the runtimes of SAT attacks on the ICs. After that, k-means clustering with the best number of clusters applied to the runtime of SAT attacks on the ICs. Then, the labels were assigned to the ICs based on the clustering results and boundary between clusters. In this study, the best number of clusters was equal 2 and the ICs were categorized into the SAT-resilient and SAT-vulnerable. In the second step, the CNF representation was obtained from the bench file format of each IC. Next, the scalar and vector features were extracted from the CNF representation of each IC. Furthermore, the best scalar features were selected in this step by SFFS method. In the third step, a dual-input GNN model called CNF-NET was trained, validated, and tested, respectively by the scalar and vector features and the labels of the ICs. </p>



## Qiuck Start 
Our codes are evaluated under python 3.7

## How to run
1. Extract the IC_Dataset.rar in the directory of files. 
2. Run Data_Generator.py to generate data and targets.
3. Run Kmeans_elbow_code.py to determine the best number of clusters (In the experiments, the best number of clutsers was 2)   
4. Run Kmeans_code.py (This code is used for investing the threshold values between clusters and formulating the mapping labeling function) 
5. Run SFS_code.py to select the best features. 
6. Run Run_iterate.py to train, validate, and test CNF-NET for classifying the circuit de-obfuscation runtime
