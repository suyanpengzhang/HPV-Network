#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:28:17 2023

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


# Replace this with the path to your actual license file
path_to_license_file = '/Users/suyanpengzhang/gurobi.lic'

# Set the environment variable
os.environ['GRB_LICENSE_FILE'] = path_to_license_file

# Now you can initialize the Gurobi environment
try:
    # Creating an environment object will automatically look for a license file
    env = Env()
    # Continue with your Gurobi model setup and optimization...
except Exception as e:
    print(f"An error occurred: {e}")



warnings.filterwarnings('ignore')
def social_connectivity(i,j):
    #i influence j
    # assume sec6 refelects how you can be influenced by others
    if hpvdata['sec6_q67'][j] < 0 :
        ans = np.nanmean(hpvdata[hpvdata['sec6_q67']>0]['sec6_q67'])
    else:
        ans = hpvdata['sec6_q67'][j]
    trustj = (ans - min_trust)/(max_trust-min_trust)
    if hpvdata['sec5_q61'][j] < 0 :
        ans = np.nanmean(hpvdata[hpvdata['sec5_q61']>0]['sec5_q61'])
    else:
        ans = hpvdata['sec5_q61'][j]
    talki = (ans - min_talk)/(max_talk-min_talk)
    if hpvdata['sec5_q62'][j] < 0 :
        ans = np.nanmean(hpvdata[hpvdata['sec5_q62']>0]['sec5_q62'])
    else:
        ans = hpvdata['sec5_q62'][j]
    talki2 = (ans - min_talk2)/(max_talk2-min_talk2)
    if talki2+talki -trustj >=0 or talki2+ talki -trustj <0 :
        return  1+ talki2 +  talki - trustj
    return 1

##read data
file_path = 'hpvdata.csv'
hpvdata = pd.read_csv(file_path)
hpvdata = hpvdata.dropna(subset=['HPV_VAX_attitu_s35'])
#compute for normalization
max_trust = 4
min_trust = 1   
max_talk = 1
min_talk = 0
max_talk2 = 3
min_talk2 = 1
max_stub = 4
min_stub = 1

num_household = len(hpvdata)

# =============================================================================
# Network = HPVN.HPV_network(num_household, 7, None) #7 is num edge attached
# Network.generate_BAGraph()
# #attitides
# id_ = [i for i in hpvdata.index]
# attitudes = [hpvdata['HPV_VAX_attitu_s35'][i] for i in hpvdata.index]
# # =============================================================================
# # stub1 = [hpvdata['sec8_q91'][i] if hpvdata['sec8_q91'][i]>0 else 2.5 for i in hpvdata.index]
# # =============================================================================
# stub2 = [(hpvdata['sec8_q92'][i]-min_stub)/(max_stub-min_stub) if hpvdata['sec8_q92'][i]>0 else (np.nanmean(hpvdata[hpvdata['sec8_q92']>0]['sec8_q92'])-min_stub)/(max_stub-min_stub) for i in hpvdata.index]
# # =============================================================================
# # listen1 = [hpvdata['sec8_q89'][i] if hpvdata['sec8_q89'][i]>0 else 2.5 for i in hpvdata.index]
# # listen2 = [hpvdata['sec8_q93'][i] if hpvdata['sec8_q93'][i]>0 else 2.5 for i in hpvdata.index]
# # =============================================================================
# good_listen_score =[stub2[i]for i in range(len(stub2))]
# # the larger the number is, less stubborn
# data = { 'id':id_, 'initial attitude': attitudes,'current attitude': attitudes, 'listen_score':good_listen_score}
# df = pd.DataFrame(data=data)
# Network.add_attributes_to_nodes(df)
# Network.add_colors([12,24])
# Network.generate_di_graph(social_connectivity)
# Network.normalize_edge_weights()
# =============================================================================
with open("network_example/network1.pkl", "rb") as file:
    Network = pickle.load(file)
#optimization
# =============================================================================
# with open("simple_net5.pkl", "rb") as file:
#     Network = pickle.load(file)
# =============================================================================

Levels = {}
edges = list(Network.G.edges)
nodes = list(Network.G.nodes)
pos_thre = 12
neg_thre = 24

def edges_to_adjacency_list(edges):
    graph = {}
    for (src, dst) in edges:
        if src not in graph:
            graph[src] = []
        if dst not in graph:
            graph[dst] = []
        graph[src].append(dst)
        graph[dst].append(src)  # Assuming it's an undirected graph
    for i in graph:
        graph[i] = list(set(graph[i]))
    return graph

def find_routes(graph, start, path=[]):
    path = path + [start]
    if len(path) == len(graph):
        return [path]
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_routes(graph, node, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


for node in nodes:
    if Network.G.nodes[node]['initial attitude']<12 or Network.G.nodes[node]['initial attitude']>=24:
        Levels[node] = 1
    else:
        Levels[node] = 0
for i in Levels:
    if Levels[i]==1:
        if i == 999:
            graph = edges_to_adjacency_list(list(Network.G.edges))
            all_routes = find_routes(graph, i)
            #print(all_routes)
        
print(len(edges))

max_depth = 0
p = nx.shortest_path(Network.G)
for i in range(len(Network.G.nodes)):
    po = [len(p[i][j]) for j in p[i]]
    if max(po)>max_depth:
        max_depth = max(po)

num_nodes = len(Network.G.nodes)
eps = 0.0001*np.ones(num_nodes)

cost_vector = np.zeros(num_nodes)
for i in Network.G.nodes:
    cost_vector[i] = Network.G.nodes[i]['initial attitude']*10
    
budget = 0
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

edge = np.zeros((num_nodes,num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        if (i,j) in Network.G.edges:
            edge[i,j] = Network.G.edges[(i,j)]['weight']

threshold_pos = 10
threshold_neg = -1
lambda_ = 0.6

T_plus = np.zeros(num_nodes)
T_minus= np.zeros(num_nodes)

T = 15+1
for i in range(num_nodes):
    T_plus[i] = threshold_pos-lambda_*threshold_pos*Network.G.nodes[i]['listen_score']
    T_minus[i] = threshold_neg+lambda_*threshold_neg*Network.G.nodes[i]['listen_score']


try:
    # Create a new model
    lm = gp.Model("lm")

    # Create variables
    
    x = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="x") 
# =============================================================================
#     a0plus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a+0_") 
#     a0minus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a-0_") 
#     a1plus = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a+1") 
#     a1minus = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a-1") 
#     a1plus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a+1_") 
#     a1minus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a-1_") 
#     a2plus = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a+2") 
#     a2minus = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a-2") 
#     a2plus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a+2_") 
#     a2minus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a-2_") 
#     a3plus = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a+3") 
#     a3minus = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a-3") 
#     a3plus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a+3_") 
#     a3minus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a-3_") 
#     a4plus = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a+4") 
#     a4minus = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a-4") 
#     a4plus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a+4_") 
#     a4minus_ = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="a-4_") 
# =============================================================================
    atplus = lm.addMVar((num_nodes,T),vtype=GRB.BINARY, name="a+t") 
    atminus = lm.addMVar((num_nodes,T),vtype=GRB.BINARY, name="a-t") 
    atplus_ = lm.addMVar((num_nodes,T),vtype=GRB.BINARY, name="a+t_") 
    atminus_ = lm.addMVar((num_nodes,T),vtype=GRB.BINARY, name="a-t_") 
    lm.setObjective(atplus_[:,T-1].sum(),GRB.MAXIMIZE)
    #lm.setObjective(a1minus_.sum(),GRB.MINIMIZE)

    ##initial
# =============================================================================
#     for i in range(2):
#         x[i].start = 1
# =============================================================================
        #lm.addConstr(x[i] == resultsx['value'][i])
    #upper bound on x
# =============================================================================
#     lm.addConstr(x[0]==1)
#     lm.addConstr(x[58]==1)
#     lm.addConstr(x[270]==1)
#     lm.addConstr(x[348]==1)
#     lm.addConstr(x[408]==1)
#     lm.addConstr(x[557]==1)
# =============================================================================
    lm.addConstr(np.transpose(cost_vector)@x<=budget)
    lm.addConstr(x+a0plus == atplus_[:,0])
    lm.addConstr(atminus_[:,0]>=a0minus-x)
    lm.addConstr(3*(atminus_[:,0]-1) <= a0minus-x-eps)
    for  t in range(1,T):
        lm.addConstr(100*(atplus[:,t]-1) <= np.transpose(edge)@atplus_[:,t-1]-np.transpose(edge)@atminus_[:,t-1]-T_plus)
        lm.addConstr(100*atplus[:,t]>=np.transpose(edge)@atplus_[:,t-1]-np.transpose(edge)@atminus_[:,t-1]-T_plus+eps)
        lm.addConstr(100*(atminus[:,t]-1) <= -np.transpose(edge)@atplus_[:,t-1]+np.transpose(edge)@atminus_[:,t-1]+T_minus)
        lm.addConstr(100*atminus[:,t]>=-np.transpose(edge)@atplus_[:,t-1]+np.transpose(edge)@atminus_[:,t-1]+T_minus+eps)
        lm.addConstr(3*atplus_[:,t]>=atplus[:,t]-atminus[:,t]+atplus_[:,t-1])
        lm.addConstr(3*(atplus_[:,t]-1)<=atplus[:,t]-atminus[:,t]+atplus_[:,t-1]-eps)
        lm.addConstr(3*atminus_[:,t]>=-atplus[:,t]+atminus[:,t]+atminus_[:,t-1])
        lm.addConstr(3*(atminus_[:,t]-1)<=-atplus[:,t]+atminus[:,t]+atminus_[:,t-1]-eps)
# =============================================================================
#     #t=1
#     lm.addConstr(100*(a1plus-1) <= np.transpose(edge)@a0plus_-np.transpose(edge)@a0minus_-T_plus)
#     lm.addConstr(100*a1plus>=np.transpose(edge)@a0plus_-np.transpose(edge)@a0minus_-T_plus+eps)
#     lm.addConstr(100*(a1minus-1) <= -np.transpose(edge)@a0plus_+np.transpose(edge)@a0minus_+T_minus)
#     lm.addConstr(100*a1minus>=-np.transpose(edge)@a0plus_+np.transpose(edge)@a0minus_+T_minus+eps)
#     lm.addConstr(3*a1plus_>=a1plus-a1minus+a0plus_)
#     lm.addConstr(3*(a1plus_-1)<=a1plus-a1minus+a0plus_-eps)
#     lm.addConstr(3*a1minus_>=-a1plus+a1minus+a0minus_)
#     lm.addConstr(3*(a1minus_-1)<=-a1plus+a1minus+a0minus_-eps)
#     #t=2
#     lm.addConstr(100*(a2plus-1) <= np.transpose(edge)@a1plus_-np.transpose(edge)@a1minus_-T_plus)
#     lm.addConstr(100*a2plus>=np.transpose(edge)@a1plus_-np.transpose(edge)@a1minus_-T_plus+eps)
#     lm.addConstr(100*(a2minus-1) <= -np.transpose(edge)@a1plus_+np.transpose(edge)@a1minus_+T_minus)
#     lm.addConstr(100*a2minus>=-np.transpose(edge)@a1plus_+np.transpose(edge)@a1minus_+T_minus+eps)
#     lm.addConstr(3*a2plus_>=a2plus-a2minus+a1plus_)
#     lm.addConstr(3*(a2plus_-1)<=a2plus-a2minus+a1plus_-eps)
#     lm.addConstr(3*a2minus_>=-a2plus+a2minus+a1minus_)
#     lm.addConstr(3*(a2minus_-1)<=-a2plus+a2minus+a1minus_-eps)
#     #t=3
#     lm.addConstr(100*(a3plus-1) <= np.transpose(edge)@a2plus_-np.transpose(edge)@a2minus_-T_plus)
#     lm.addConstr(100*a3plus>=np.transpose(edge)@a2plus_-np.transpose(edge)@a2minus_-T_plus+eps)
#     lm.addConstr(100*(a3minus-1) <= -np.transpose(edge)@a2plus_+np.transpose(edge)@a2minus_+T_minus)
#     lm.addConstr(100*a3minus>=-np.transpose(edge)@a2plus_+np.transpose(edge)@a2minus_+T_minus+eps)
#     lm.addConstr(3*a3plus_>=a3plus-a3minus+a2plus_)
#     lm.addConstr(3*(a3plus_-1)<=a3plus-a3minus+a2plus_-eps)
#     lm.addConstr(3*a3minus_>=-a3plus+a3minus+a2minus_)
#     lm.addConstr(3*(a3minus_-1)<=-a3plus+a3minus+a2minus_-eps)
#     #t=4
#     lm.addConstr(100*(a4plus-1) <= np.transpose(edge)@a3plus_-np.transpose(edge)@a3minus_-T_plus)
#     lm.addConstr(100*a4plus>=np.transpose(edge)@a3plus_-np.transpose(edge)@a3minus_-T_plus+eps)
#     lm.addConstr(100*(a4minus-1) <= -np.transpose(edge)@a3plus_+np.transpose(edge)@a3minus_+T_minus)
#     lm.addConstr(100*a4minus>=-np.transpose(edge)@a3plus_+np.transpose(edge)@a3minus_+T_minus+eps)
#     lm.addConstr(3*a4plus_>=a4plus-a4minus+a3plus_)
#     lm.addConstr(3*(a4plus_-1)<=a4plus-a4minus+a3plus_-eps)
#     lm.addConstr(3*a4minus_>=-a4plus+a4minus+a3minus_)
#     lm.addConstr(3*(a4minus_-1)<=-a4plus+a4minus+a3minus_-eps)
# =============================================================================
    # Optimize model
    lm.setParam('TimeLimit', 10)
    lm.Params.Threads = 18
    lm.Params.OutputFlag = 1
    lm.Params.LogToConsole = 1
    lm.optimize()
    sol = []
    count=0
    print('LP:')
    for v in lm.getVars():
        if count<num_nodes:
            if v.X == 1:
                print('%s %g' % (v.VarName, v.X))
                sol.append(count)
        count+=1
    print('Obj: %g' % lm.ObjVal)
    
except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
print('********************')    
print('Selections')
for i in sol:
    print('attitude: ',Network.G.nodes[i]['initial attitude'])
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
    print('linked to pos: ',countplus)
    print('linked to neg: ',countminus)
    print('linked to neu: ',countneu)
print('********************')
Network.run_linear_threshold_model(lambda_ = 0.6,threshold_pos=10,threshold_neg=-1,inital_threshold=[12,24],time_periods=10)

np.transpose(edge[:,187])@a0plus-np.transpose(edge[:,187])@a0minus-T_plus[187]
t=0
for Gs in Network.LTM:
    print('********************')
    print('time',t)
    t+=1
    ls = np.array([Gs.nodes.data('status')[i] for i in Gs.nodes])
    mask_pos = np.where(ls==1)
    mask_neg = np.where(ls==-1)
    age = np.array([Gs.nodes.data('current attitude')[i] for i in Gs.nodes])
    pos_age = age[mask_pos]
    neg_age = age[mask_neg]
    print('Num pos',len(mask_pos[0]))
    print('Num negative',len(mask_neg[0]))
    print('Mean current att among pos',np.mean(pos_age))
    print('Mean current att among neg',np.mean(neg_age))
    #node_colors = [Gs.nodes.data('color')[i] for i in range(num_household)]
    #nx.draw_networkx(Gs, with_labels=True, node_color=node_colors)
    plt.show()

