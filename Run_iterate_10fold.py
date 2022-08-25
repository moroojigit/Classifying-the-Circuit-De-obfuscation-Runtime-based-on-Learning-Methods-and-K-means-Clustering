# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:29:05 2022

@author: Mahdi Orooji
"""

import time
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from operator import itemgetter
from torch.utils.data import DataLoader
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from model import CNFNet
from util import GraphDataset, chunks, print_network, plot_metric, read_ic, read_ic_10fold

from scipy.io import savemat, loadmat

def Evaluate_CM(L_test,Predict):
    NTP = 0 
    NTN = 0 
    NFP = 0 
    NFN = 0 
    for i in range(len(Predict)):
        if (Predict[i]==1)and(L_test[i]==1):
            NTP = NTP + 1
        elif (Predict[i]==0)and(L_test[i]==0):
            NTN = NTN + 1 
        elif (Predict[i]==1)and(L_test[i]==0):
            NFP = NFP + 1 
        else:
            NFN = NFN + 1 
    Accuracy = (NTP+NTN)/(NTP+NTN+NFP+NFN)
    Sensitivity = NTP/(NTP+NFN)
    try:
        Specificity = NTN/(NTN+NFP)
    except:
        Specificity = np.nan
    F1 = (2*NTP)/((2*NTP)+NFP+NFN)
    Loss = 1 - Accuracy
    return Accuracy, Sensitivity, Specificity, F1, Loss 

def Confision_Matrix(All_test_predic,All_test_label):
    ACs = []
    SEs = []
    SPs = []
    F1s = []
    NPs = [] 
    NNs = []
    for k in range(len(All_test_label)): 
        L_test = All_test_label[k]
        Predict = All_test_predic[k]
        NPs.append(np.sum(np.asarray(L_test)==1)) 
        NNs.append(np.sum(np.asarray(L_test)==0)) 
        AC, SE, SP, F1, _ = Evaluate_CM(Predict,L_test)
        ACs.append(AC)
        SEs.append(SE)
        SPs.append(SP)
        F1s.append(F1)
    return ACs, SEs, SPs, F1s, NPs, NNs

def Prediction(output,model): 
    _, predicted = torch.max(output.data, 1)
    return predicted    

def Obtain_Accuracy(output,labels,model):
    correct = 0
    total = 0
    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    Accuracy = correct/total 
    return Accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=8, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='c499', help='Dataset name')

#CircuitNames = ['74181','74182','74283','74L85','c1196','c1238','c1355','c17',
# 'c2670','c3540','c432','c499','c5315','c6288','c7552','c880','s1196','s1196a',
# 's1238','s1238a']
CircuitNames = ['c1355','c2670','c3540','c5315','c6288','c7552']
CircuitNames = [CircuitNames[1]]

args = parser.parse_args()
args.cuda = True

# Loading Selected Features indexes 
Dict_SF = loadmat("SF.mat")
SF = Dict_SF["Selected_Features"]

# Training setting
args.epochs = 25
args.num_feat = SF.size
args.batch_size = 32

# FC setting
args.energy_input_dim = SF.size + 7 
Number_of_scaler_features = 40 
#m1 = 16780 
m2 = 59830

Num_iter = 1
All_test_acc = []
All_test_loss = []
All_test_predic = []
All_test_label = []

inc_feat, feats, times = read_ic_10fold(CircuitNames,Number_of_scaler_features,
                                        m2,True,SF)

kf = KFold(n_splits=2)
kf.get_n_splits(feats,times)

ACs = []
SEs = []
SPs = []
F1s = []
NPs = [] 
NNs = []


i=0
for train_index, test_index in kf.split(feats,times):
    i =  i + 1 
    
    inc_feat_tr, inc_feat_te, feat_tr, feat_te, labels_tr, labels_te = train_test_split([inc_feat[p] for p in train_index], feats[train_index,:],
                     times[train_index], test_size=0.1)
    
    model = CNFNet(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print_network(model)
    
    #cri = nn.CrossEntropyLoss(weight=torch.Tensor([float((times==1).sum()/(times==0).sum()) ,1]))
    cri = nn.CrossEntropyLoss()
    
    Tr_incides = np.arange(len(inc_feat_tr))
    
    Tr_incides = torch.LongTensor(Tr_incides)
    test_index = torch.LongTensor(test_index)
    
    graph_loader = DataLoader(GraphDataset(Tr_incides[:int(len(Tr_incides) / args.batch_size) * args.batch_size]),
                              batch_size=args.batch_size, shuffle=True)
    
    train_loss = []
    eval_loss = []
    train_acc = []
    val_acc = []
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True
    
    # Training Model
    for epoch in range(args.epochs):
        for step, ids in enumerate(graph_loader):
            
            t = time.time()
            model.train()
            output = model(itemgetter(*ids)(inc_feat_tr), feat_tr[ids])
            loss_train = cri(output, labels_tr[ids])
                
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            print('Epoch: {:02d}/{:04d}'.format(epoch, step + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'time: {:.4f}s'.format(time.time() - t), end='\n')

            if step % 10 == 0:
                model.eval()
                #val_ids = list(chunks(val_num, args.batch_size))[:-1]
                output_eval = [model(inc_feat_te, feat_te)]
                
                #output_eval = [model(itemgetter(*_)(inc_feat), feats[_]) for _ in val_ids]
                loss_val = np.mean([cri(output_eval[0][_], labels_te[_]).item() for _ in range(len(labels_te))])
                
                #loss_val = np.mean([cri(output_eval[_], times[val_ids[_]]).item() for _ in range(len(val_ids))])
                print("Eval loss: {}".format(loss_val), end='\n')
                train_acc.append(Obtain_Accuracy(output,labels_tr[ids],model))
                
                val_acc.append(Obtain_Accuracy(output_eval[0],labels_te,model))
                train_loss.append(loss_train.item())
                eval_loss.append(loss_val.item())
                
        # print training info
    plot_metric(range(len(train_loss)), train_loss, eval_loss, '{}_train_{}'.format('loss',i),
                '{}_eval_{}'.format('loss',i))

    plot_metric(range(len(train_acc)), train_acc, val_acc, '{}_train_{}'.format('Acc',i),
                '{}_eval_{}'.format('Acc',i))

    print("Optimization Finished!")
        
    model.eval()
    test_ids = list(chunks(test_index, args.batch_size))
    output_test = []
    output_test = [model(itemgetter(*_)(inc_feat), feats[_]) for _ in test_ids]
    loss_val = np.mean([cri(output_test[_], times[test_ids[_]]).item() for _ in range(len(test_ids))])
    print("Test loss: {}".format(loss_val))
    test_acc = Obtain_Accuracy(output_test[0],times[test_ids[0]],model)
    test_pred = Prediction(output_test[0],model)
    
    NPs.append(np.sum(np.asarray(times[test_ids[0]].tolist())==1)) 
    NNs.append(np.sum(np.asarray(times[test_ids[0]].tolist())==0)) 
    
    AC, SE, SP, F1, _ = Evaluate_CM(test_pred.tolist(),times[test_ids[0]].tolist())
    
    ACs.append(AC)
    SEs.append(SE)
    SPs.append(SP)
    F1s.append(F1)
    
    print(test_acc)
    All_test_acc.append(test_acc)
    All_test_loss.append(loss_val)

    
All_test_acc = np.asarray(All_test_acc)
Mean_All_test_acc = np.mean(All_test_acc)

All_test_loss = np.asarray(All_test_loss)
Mean_All_test_loss = np.mean(All_test_loss)

Mydic = {"All_test_acc" : All_test_acc, "Mean_All_test_acc" : Mean_All_test_acc, 
         "Mean_All_test_loss":Mean_All_test_loss, "All_test_loss" : All_test_loss,'ACs':np.asarray(ACs),
         'SEs':np.asarray(SEs), 'SPs':np.asarray(SPs), 'F1s':np.asarray(F1s)}

savemat('results.mat',Mydic)





