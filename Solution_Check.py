#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:16:48 2023

@author: suyanpengzhang
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import HPVnetwork as HPVN
import warnings
import seaborn as sns
import gurobipy as gp
from gurobipy import GRB
import pickle
import os
from gurobipy import Env, GRB
from collections import deque
from numpy import linalg as LA

with open("network_example/network1.pkl", "rb") as file:
    Network = pickle.load(file)
with open("sols/sol_1000.pkl", "rb") as file:
    sol1000 = pickle.load(file)
with open("sols/sol_2000.pkl", "rb") as file:
    sol2000 = pickle.load(file)
with open("sols/sol_3000.pkl", "rb") as file:
    sol3000 = pickle.load(file)
with open("sols/sol_4000.pkl", "rb") as file:
    sol4000 = pickle.load(file)
with open("sols/sol_5000.pkl", "rb") as file:
    sol5000 = pickle.load(file)
with open("sols/sol_6000.pkl", "rb") as file:
    sol6000 = pickle.load(file)
with open("sols/sol_7000.pkl", "rb") as file:
    sol7000 = pickle.load(file)
initial_status = [Network.G.nodes[i]['initial attitude'] for i in Network.G.nodes]
for i in range(len(initial_status)):
    if initial_status[i]>=24:
        initial_status[i]=-1
    elif initial_status[i]<12:
        initial_status[i]=1
    else:
        initial_status[i]=0
a0plus = np.array([1 if i==1 else 0 for i in initial_status])
a0minus = np.array([1 if i==-1 else 0 for i in initial_status])
num_nodes = len(Network.G.nodes)
threshold_pos = 10
threshold_neg = -1
lambda_ = 0.6

T_plus = np.zeros(num_nodes)
T_minus= np.zeros(num_nodes)

T = 8



for i in range(num_nodes):
    T_plus[i] = threshold_pos-lambda_*threshold_pos*Network.G.nodes[i]['listen_score']
    T_minus[i] = threshold_neg+lambda_*threshold_neg*Network.G.nodes[i]['listen_score']
edge = np.zeros((num_nodes,num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        if (i,j) in Network.G.edges:
            edge[i,j] = Network.G.edges[(i,j)]['weight']
eigenvalues, eigenvectors = LA.eig(edge)
threshold_pos = 10
threshold_neg = -1
lambda_ = 0.6

T_plus = np.zeros(num_nodes)
T_minus= np.zeros(num_nodes)
print('1000\n',sol1000)
plus_ = []
minus_ = []
neu_ = []
attitude_ = []
edge_weights_= []
eigenvalues_ = []
for i in sol1000:
    countplus = 0
    countminus = 0
    countneu = 0
    for j in Network.G.nodes:
        if (i,j) in Network.G.edges:
            if a0plus[j]==1:
                countplus +=1
            elif a0minus[j] == 1:
                countminus +=1
            else:
                countneu +=1
    plus_.append(countplus)
    minus_.append(countminus)
    neu_.append(countneu)
    attitude_.append(Network.G.nodes[i]['initial attitude'])
    edge_weights_.append(sum(edge[i]))
    eigenvalues_.append(eigenvalues[i])

print('avg links to pos',np.mean(plus_))
print('avg links to minus',np.mean(minus_))
print('avg links to neutral',np.mean(neu_))
print('avg attitude score',np.mean(attitude_))
print('avg outgoing edgeweights',np.mean(edge_weights_))
print('avg eigenvalues',np.mean(eigenvalues_))
print('avg score',np.mean(np.array(edge_weights_)/np.array(attitude_)))

print('2000\n',sol2000)

plus_ = []
minus_ = []
neu_ = []
attitude_ = []
edge_weights_= []
eigenvalues_ = []
for i in sol2000:
    countplus = 0
    countminus = 0
    countneu = 0
    for j in Network.G.nodes:
        if (i,j) in Network.G.edges:
            if a0plus[j]==1:
                countplus +=1
            elif a0minus[j] == 1:
                countminus +=1
            else:
                countneu +=1
    plus_.append(countplus)
    minus_.append(countminus)
    neu_.append(countneu)
    attitude_.append(Network.G.nodes[i]['initial attitude'])
    edge_weights_.append(sum(edge[i]))
    eigenvalues_.append(eigenvalues[i])

print('avg links to pos',np.mean(plus_))
print('avg links to minus',np.mean(minus_))
print('avg links to neutral',np.mean(neu_))
print('avg attitude score',np.mean(attitude_))
print('avg outgoing edgeweights',np.mean(edge_weights_))
print('avg eigenvalues',np.mean(eigenvalues_))
print('avg score',np.mean(np.array(edge_weights_)/np.array(attitude_)))

print('3000\n',sol3000)

plus_ = []
minus_ = []
neu_ = []
attitude_ = []
edge_weights_= []
eigenvalues_ = []
for i in sol3000:
    countplus = 0
    countminus = 0
    countneu = 0
    for j in Network.G.nodes:
        if (i,j) in Network.G.edges:
            if a0plus[j]==1:
                countplus +=1
            elif a0minus[j] == 1:
                countminus +=1
            else:
                countneu +=1
    plus_.append(countplus)
    minus_.append(countminus)
    neu_.append(countneu)
    attitude_.append(Network.G.nodes[i]['initial attitude'])
    edge_weights_.append(sum(edge[i]))
    eigenvalues_.append(eigenvalues[i])

print('avg links to pos',np.mean(plus_))
print('avg links to minus',np.mean(minus_))
print('avg links to neutral',np.mean(neu_))
print('avg attitude score',np.mean(attitude_))
print('avg outgoing edgeweights',np.mean(edge_weights_))
print('avg eigenvalues',np.mean(eigenvalues_))
print('avg score',np.mean(np.array(edge_weights_)/np.array(attitude_)))

print('4000\n',sol4000)

plus_ = []
minus_ = []
neu_ = []
attitude_ = []
edge_weights_= []
eigenvalues_ = []
for i in sol4000:
    countplus = 0
    countminus = 0
    countneu = 0
    for j in Network.G.nodes:
        if (i,j) in Network.G.edges:
            if a0plus[j]==1:
                countplus +=1
            elif a0minus[j] == 1:
                countminus +=1
            else:
                countneu +=1
    plus_.append(countplus)
    minus_.append(countminus)
    neu_.append(countneu)
    attitude_.append(Network.G.nodes[i]['initial attitude'])
    edge_weights_.append(sum(edge[i]))
    eigenvalues_.append(eigenvalues[i])

print('avg links to pos',np.mean(plus_))
print('avg links to minus',np.mean(minus_))
print('avg links to neutral',np.mean(neu_))
print('avg attitude score',np.mean(attitude_))
print('avg outgoing edgeweights',np.mean(edge_weights_))
print('avg eigenvalues',np.mean(eigenvalues_))
print('avg score',np.mean(np.array(edge_weights_)/np.array(attitude_)))

print('5000\n',sol5000)

plus_ = []
minus_ = []
neu_ = []
attitude_ = []
edge_weights_= []
eigenvalues_ = []
for i in sol5000:
    countplus = 0
    countminus = 0
    countneu = 0
    for j in Network.G.nodes:
        if (i,j) in Network.G.edges:
            if a0plus[j]==1:
                countplus +=1
            elif a0minus[j] == 1:
                countminus +=1
            else:
                countneu +=1
    plus_.append(countplus)
    minus_.append(countminus)
    neu_.append(countneu)
    attitude_.append(Network.G.nodes[i]['initial attitude'])
    edge_weights_.append(sum(edge[i]))
    eigenvalues_.append(eigenvalues[i])

print('avg links to pos',np.mean(plus_))
print('avg links to minus',np.mean(minus_))
print('avg links to neutral',np.mean(neu_))
print('avg attitude score',np.mean(attitude_))
print('avg outgoing edgeweights',np.mean(edge_weights_))
print('avg eigenvalues',np.mean(eigenvalues_))
print('avg score',np.mean(np.array(edge_weights_)/np.array(attitude_)))
print('6000\n',sol6000)

plus_ = []
minus_ = []
neu_ = []
attitude_ = []
edge_weights_= []
eigenvalues_ = []
for i in sol6000:
    countplus = 0
    countminus = 0
    countneu = 0
    for j in Network.G.nodes:
        if (i,j) in Network.G.edges:
            if a0plus[j]==1:
                countplus +=1
            elif a0minus[j] == 1:
                countminus +=1
            else:
                countneu +=1
    plus_.append(countplus)
    minus_.append(countminus)
    neu_.append(countneu)
    attitude_.append(Network.G.nodes[i]['initial attitude'])
    edge_weights_.append(sum(edge[i]))
    eigenvalues_.append(eigenvalues[i])

print('avg links to pos',np.mean(plus_))
print('avg links to minus',np.mean(minus_))
print('avg links to neutral',np.mean(neu_))
print('avg attitude score',np.mean(attitude_))
print('avg outgoing edgeweights',np.mean(edge_weights_))
print('avg eigenvalues',np.mean(eigenvalues_))
print('avg score',np.mean(np.array(edge_weights_)/np.array(attitude_)))

print('7000\n',sol7000)

plus_ = []
minus_ = []
neu_ = []
attitude_ = []
edge_weights_= []
eigenvalues_ = []
for i in sol7000:
    countplus = 0
    countminus = 0
    countneu = 0
    for j in Network.G.nodes:
        if (i,j) in Network.G.edges:
            if a0plus[j]==1:
                countplus +=1
            elif a0minus[j] == 1:
                countminus +=1
            else:
                countneu +=1
    plus_.append(countplus)
    minus_.append(countminus)
    neu_.append(countneu)
    attitude_.append(Network.G.nodes[i]['initial attitude'])
    edge_weights_.append(sum(edge[i]))
    eigenvalues_.append(eigenvalues[i])

print('avg links to pos',np.mean(plus_))
print('avg links to minus',np.mean(minus_))
print('avg links to neutral',np.mean(neu_))
print('avg attitude score',np.mean(attitude_))
print('avg outgoing edgeweights',np.mean(edge_weights_))
print('avg eigenvalues',np.mean(eigenvalues_))
print('avg score',np.mean(np.array(edge_weights_)/np.array(attitude_)))

print('all\n')

plus_ = []
minus_ = []
neu_ = []
attitude_ = []
edge_weights_= []
eigenvalues_ = []
for i in np.where(1-a0plus==1)[0]:
    countplus = 0
    countminus = 0
    countneu = 0
    for j in Network.G.nodes:
        if (i,j) in Network.G.edges:
            if a0plus[j]==1:
                countplus +=1
            elif a0minus[j] == 1:
                countminus +=1
            else:
                countneu +=1
    plus_.append(countplus)
    minus_.append(countminus)
    neu_.append(countneu)
    attitude_.append(Network.G.nodes[i]['initial attitude'])
    edge_weights_.append(sum(edge[i]))
    eigenvalues_.append(eigenvalues[i])

print('avg links to pos',np.mean(plus_))
print('avg links to minus',np.mean(minus_))
print('avg links to neutral',np.mean(neu_))
print('avg attitude score',np.mean(attitude_))
print('avg outgoing edgeweights',np.mean(edge_weights_))
print('avg eigenvalues',np.mean(eigenvalues_))
print('avg score',np.mean(np.array(edge_weights_)/np.array(attitude_)))

# =============================================================================
# Network.run_linear_threshold_model(lambda_ = 0.6,threshold_pos=10,threshold_neg=-1,inital_threshold=[12,24],time_periods=10,x=sol7000)
# t=0
# for Gs in Network.LTM:
#     print('********************')
#     print('time',t)
#     t+=1
#     ls = np.array([Gs.nodes.data('status')[i] for i in Gs.nodes])
#     mask_pos = np.where(ls==1)
#     mask_neg = np.where(ls==-1)
#     age = np.array([Gs.nodes.data('current attitude')[i] for i in Gs.nodes])
#     pos_age = age[mask_pos]
#     neg_age = age[mask_neg]
#     print('Num pos',len(mask_pos[0]))
#     print('Num negative',len(mask_neg[0]))
#     print('Mean current att among pos',np.mean(pos_age))
#     print('Mean current att among neg',np.mean(neg_age))
#     plt.show()
# 
# =============================================================================
