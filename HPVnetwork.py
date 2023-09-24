#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:34:20 2023

@author: suyanpengzhang
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random



class HPV_network(object):
    def __init__(self,number_of_nodes,num_edge,seed):
        self.num_nodes = number_of_nodes
        self.num_edge_attached = num_edge
        self.seed = seed
        self.G= None
        self.LTM = None
    def generate_BAGraph(self):
        self.G = nx.barabasi_albert_graph(self.num_nodes,self.num_edge_attached,seed=self.seed)
        #nx.draw(self.G, with_labels=True)
        #plt.show()
    def add_attributes_to_nodes(self,attr_table):
        if len(attr_table) != len(self.G.nodes):
            print('Error: data & nodes mismatch')
            return
        for node in range(len(attr_table)):
            for atr in range(0,len(attr_table.columns)):
                self.G.nodes[node][attr_table.columns[atr]] = attr_table[attr_table.columns[atr]][node]
    def add_colors(self,thresholds):
        for node in range(len(self.G.nodes)):
            if self.G.nodes[node]['initial attitude']<thresholds[0]:
                self.G.nodes[node]['color'] = 'red'
            elif self.G.nodes[node]['initial attitude']>=thresholds[1]:   
                self.G.nodes[node]['color'] = 'blue'
            else:
                self.G.nodes[node]['color'] = 'green'
    def generate_di_graph(self,function):
        NewG = nx.DiGraph(self.G)
        for edge in NewG.edges:
            NewG[edge[0]][edge[1]]['weight'] = function(NewG.nodes[edge[0]]['id'],NewG.nodes[edge[1]]['id'])
# =============================================================================
#             if NewG.nodes[edge[0]]['initial attitude'] < 12:
#                 NewG[edge[0]][edge[1]]['weight'] = 0.5
#             elif NewG.nodes[edge[0]]['initial attitude'] >= 24:
#                 NewG[edge[0]][edge[1]]['weight'] = 0.4
#             else:
#                 NewG[edge[0]][edge[1]]['weight'] = 0.55
# =============================================================================
        self.G = NewG
# =============================================================================
#         nx.draw(self.G, with_labels=True)
#         pos = nx.random_layout(self.G)
#         edge_labels = nx.get_edge_attributes(self.G, "weight")
#         print(edge_labels)
#         nx.draw_networkx_edge_labels(self.G, pos,edge_labels=edge_labels,font_size = 10)
#         plt.show()
# =============================================================================
    def run_linear_threshold_model(self,rand,threshold_pos,threshold_neg,inital_threshold,time_periods):
        if rand == 1:
            self.LTM = []
            for node in range(len(self.G.nodes)):
                if self.G.nodes[node]['initial attitude']<inital_threshold[0]:
                    self.G.nodes[node]['status'] = 1
                    self.G.nodes[node]['color'] = 'red'
                elif self.G.nodes[node]['initial attitude']>=inital_threshold[1]:   
                    self.G.nodes[node]['status'] = -1
                    self.G.nodes[node]['color'] = 'blue'
                else:
                    self.G.nodes[node]['status'] = 0
                    self.G.nodes[node]['color'] = 'green'
            self.LTM.append(self.G)
            if threshold_pos == None:
                threshold_pos = np.random.random()
            newG = self.G.copy()
            for t in range(time_periods):
                print(t)
                neg = []
                pos = []
                for node in range(len(self.G.nodes)):
                    score = 0 
                    for outnode in range(len(self.G.nodes)):
                        if (outnode,node) in newG.edges:
                            if newG.nodes[outnode]['status']==1:
                                score += newG.edges[(outnode,node)]['weight']
                            elif newG.nodes[outnode]['status']==-1:
                                score -= newG.edges[(outnode,node)]['weight']
                    if score >= threshold_pos:
                        pos.append(node)
                    if score <= threshold_neg:
                        neg.append(node)
                for node in pos:
                    newG.nodes[node]['status'] = 1
                    newG.nodes[node]['color'] = 'red'
                for node in neg:
                    newG.nodes[node]['status'] = -1     
                    newG.nodes[node]['color'] = 'blue'
                copy_G = newG.copy()
                self.LTM.append(copy_G)





