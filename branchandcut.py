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
from numpy import linalg as LA



# Path to your actual license file
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
file_path = 'Data/hpvdata.csv'
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
for ii in range(0,1):
    network_name = 'network'+str(ii)
    with open("100_Network_Samples/"+network_name+".pkl", "rb") as file:
        Network = pickle.load(file)
    
    Levels = {}
    edges = list(Network.G.edges)
    nodes = list(Network.G.nodes)
    
                
    pos_thre = 12
    neg_thre = 24
    
    
            
    print(len(edges))
    
    
    num_nodes = len(Network.G.nodes)
    eps = 0.0001
    
    cost_vector = np.zeros(num_nodes)
    for i in Network.G.nodes:
        cost_vector[i] = Network.G.nodes[i]['initial attitude']*10
        cost_vector[i] = 100
        
    budget = 50
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
    #eigenvalues, eigenvectors = LA.eig(edge)
    threshold_pos = 8
    threshold_neg = -1
    lambda_ = 0.6
    
    T_plus = np.zeros(num_nodes)
    T_minus= np.zeros(num_nodes)
    
    T = 8
    
    
    
    for i in range(num_nodes):
        T_plus[i] = threshold_pos-lambda_*threshold_pos*Network.G.nodes[i]['listen_score']
        T_minus[i] = threshold_neg+lambda_*threshold_neg*Network.G.nodes[i]['listen_score']
    
    try:
        # Create a new model
        lm = gp.Model("lm")
    
        # Create variables
        
        x = lm.addMVar(num_nodes,vtype=GRB.BINARY, name="x") 
        atplus = lm.addMVar((num_nodes,T),vtype=GRB.BINARY, name="a+t") 
        atminus = lm.addMVar((num_nodes,T),vtype=GRB.BINARY, name="a-t") 
        atplus_ = lm.addMVar((num_nodes,T),vtype=GRB.BINARY, name="a+t_") 
        atminus_ = lm.addMVar((num_nodes,T),vtype=GRB.BINARY, name="a-t_") 
        lm.setObjective(sum(atplus_[:,T-1]),GRB.MAXIMIZE)
    # =============================================================================
    #     for i in range(num_nodes):
    #         if i in sol1000:
    #             #x[i].start = 1
    #             lm.addConstr(x[i]==1)
    # =============================================================================
    # =============================================================================
    #         if a0plus[i] == 1:
    #             lm.addConstr(x[i]==0)
    # =============================================================================
        #lm.addConstr(np.transpose(cost_vector)@x<=budget)
        lm.addConstr(sum(x)==budget)
        lm.addConstr(3*atplus_[:,0]>=a0plus+x)
        lm.addConstr(3*(atplus_[:,0]-np.ones(num_nodes))<=a0plus+x-eps)
        lm.addConstr(3*atminus_[:,0]>=a0minus-x)
        lm.addConstr(3*(atminus_[:,0]-np.ones(num_nodes)) <= a0minus-x-eps)
        for  t in range(1,T):
            lm.addConstr(1000*(atplus[:,t]-np.ones(num_nodes)) <= np.transpose(edge)@atplus_[:,t-1]-np.transpose(edge)@atminus_[:,t-1]-T_plus)
            lm.addConstr(1000*atplus[:,t]>=np.transpose(edge)@atplus_[:,t-1]-np.transpose(edge)@atminus_[:,t-1]-T_plus+eps)
            lm.addConstr(1000*(atminus[:,t]-np.ones(num_nodes)) <= -np.transpose(edge)@atplus_[:,t-1]+np.transpose(edge)@atminus_[:,t-1]+T_minus)
            lm.addConstr(1000*atminus[:,t]>=-np.transpose(edge)@atplus_[:,t-1]+np.transpose(edge)@atminus_[:,t-1]+T_minus+eps)
            lm.addConstr(3*atplus_[:,t]>=atplus[:,t]-atminus[:,t]+atplus_[:,t-1])
            lm.addConstr(3*(atplus_[:,t]-np.ones(num_nodes))<=atplus[:,t]-atminus[:,t]+atplus_[:,t-1]-eps)
            lm.addConstr(3*atminus_[:,t]>=-atplus[:,t]+atminus[:,t]+atminus_[:,t-1])
            lm.addConstr(3*(atminus_[:,t]-np.ones(num_nodes))<=-atplus[:,t]+atminus[:,t]+atminus_[:,t-1]-eps)
        # Optimize model
        lm.setParam('TimeLimit', 1800)
        lm.Params.Threads = 18
        lm.Params.OutputFlag = 1
        lm.Params.LogToConsole = 1
        def myheuristic(model, where):
            if where == GRB.Callback.MIPNODE:
                # Check if it is a feasible solution node
                if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                    # Get the current relaxation solution
                    node_rel_vals = model.cbGetNodeRel(x)
                    #print(node_rel_vals)
                    # Implement your heuristic logic here
                    new_val  = np.round(node_rel_vals,0)
                    #new_vals[new_plus] = 1  # Set z to 1
                    model.cbSetSolution(x, new_val)
                        # Try to use the current node's solution
                    model.cbUseSolution()
        #lm.optimize(myheuristic)
        #lm.Params.MIPFocus = 0
        #lm.Params.NoRelHeurTime = 30
        lm.setParam('TimeLimit', 60)
        lm.optimize()
        sol = []
        count=0
        print('LP:')
        for v in lm.getVars():
            if count<num_nodes:
                if v.X == 1:
                    sol.append(count)     
            count+=1
        print('Obj: %g' % lm.ObjVal)
        
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
    print('********************')    
# =============================================================================
#     with open("100_Network_Sols/sol_50_rho_0"+network_name+".pkl", 'wb') as f:
#         pickle.dump(sol, f)
# =============================================================================
