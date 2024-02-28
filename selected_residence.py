#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:18:19 2024

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats

with open("100_Network_Samples/network0.pkl", "rb") as file:
    Network = pickle.load(file)

file_path = 'Data/hpvdata.csv'
hpvdata = pd.read_csv(file_path)
hpvdata = hpvdata.dropna(subset=['HPV_VAX_attitu_s35'])



def ttest(str_name,sol_file):
    var_ = [i for i in hpvdata[str_name].values]
    with open(sol_file, "rb") as file:
        soluniform = pickle.load(file)
    selected = []
    unselected =[]
    for i in range(len(var_)):
        if i in soluniform:
            if var_[i]>0:
                selected.append(var_[i])
        else:
            if var_[i]>0:
                unselected.append(var_[i])
    #print(str_name)
    test = stats.ttest_ind(selected, unselected)
    if test.pvalue<0.01:
        print(stats.ttest_ind(selected, unselected))
        print(np.mean(selected))
        print(np.mean(unselected))
for i in range(60):
    ttest('HPV_VAX_attitu_s35',"100_Network_Sols/sol_50_network"+str(i)+".pkl")
# =============================================================================
# income = [i for i in hpvdata['sec1_q8'].values]
# residence = [i for i in hpvdata['sec1_q9'].values]
# with open("100_Network_Sols/sol_50_network0.pkl", "rb") as file:
#     soluniform = pickle.load(file)
# selected_income =[]
# unselected_income =[]
# selected_residence =[]
# unselected_residence =[]
# 
# for i in range(len(income)):
#     if i in soluniform:
#         if income[i]>0:
#             selected_income.append(income[i])
#             selected_residence.append(residence[i])
#     else:
#         if income[i]>0:
#             unselected_income.append(income[i])
#             unselected_residence.append(residence[i])
# print('income')
# print(stats.ttest_ind(selected_income, unselected_income))
# print('residence')
# print(stats.ttest_ind(selected_residence ,unselected_residence))
# =============================================================================
