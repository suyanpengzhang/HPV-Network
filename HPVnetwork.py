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
    def add_colors(self):
        for node in range(len(self.G.nodes)):
            if self.G.nodes[node]['initial attitude']<12:
                self.G.nodes[node]['color'] = 'red'
            elif self.G.nodes[node]['initial attitude']>=24:   
                self.G.nodes[node]['color'] = 'blue'
            else:
                self.G.nodes[node]['color'] = 'green'
    def generate_di_graph(self,rand):
        if rand == 1:
            NewG = nx.DiGraph(self.G)
            for edge in NewG.edges:
                if NewG.nodes[edge[0]]['initial attitude'] < 12:
                    NewG[edge[0]][edge[1]]['weight'] = 0.5
                elif NewG.nodes[edge[0]]['initial attitude'] >= 24:
                    NewG[edge[0]][edge[1]]['weight'] = 0.4
                else:
                    NewG[edge[0]][edge[1]]['weight'] = 0.55
            self.G = NewG
# =============================================================================
#         nx.draw(self.G, with_labels=True)
#         pos = nx.random_layout(self.G)
#         edge_labels = nx.get_edge_attributes(self.G, "weight")
#         print(edge_labels)
#         nx.draw_networkx_edge_labels(self.G, pos,edge_labels=edge_labels,font_size = 10)
#         plt.show()
# =============================================================================
    def run_linear_threshold_model(self,rand,threshold_pos,threshold_neg,inital_pos,inital_neg, num_initial_pos,num_initial_neg,time_periods):
        if rand == 1:
            self.LTM = []
# =============================================================================
#             inital = random.sample(list(self.G.nodes),k=num_initial_pos+num_initial_neg)
#             #print(inital)
#             if inital_pos == []:
#                 inital_pos = inital[:num_initial_pos]
#             if inital_neg == []:
#                 inital_neg = inital[num_initial_pos:num_initial_pos+num_initial_neg]
#             for node in range(len(self.G.nodes)):
#                 if node in inital_pos:
#                     self.G.nodes[node]['status'] = 1
#                 elif node in inital_neg:
#                     self.G.nodes[node]['status'] = -1
#                 else:
#                     self.G.nodes[node]['status'] = 0
# =============================================================================
            for node in range(len(self.G.nodes)):
                if self.G.nodes[node]['initial attitude']<12:
                    self.G.nodes[node]['status'] = 1
                    self.G.nodes[node]['color'] = 'red'
                elif self.G.nodes[node]['initial attitude']>=24:   
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
                    '''
                    if newG.nodes[node]['status']==1:
                        continue
                    elif newG.nodes[node]['status']==-1:
                        continue
                    else:
                        '''
                    score = 0 
                    #flag = False
                    for outnode in range(len(self.G.nodes)):
                        if (outnode,node) in newG.edges:
                            if newG.nodes[outnode]['status']==1:
                                #flag = True
                                score += newG.edges[(outnode,node)]['weight']
                            elif newG.nodes[outnode]['status']==-1:
                                #flag = True
                                score -= newG.edges[(outnode,node)]['weight']
# =============================================================================
#                     if node == 0:
#                         print(score)
# =============================================================================
                    if score >= threshold_pos:
                        #print(score)
                        pos.append(node)
                        #newG.nodes[node]['status'] = 1
                    if score <= threshold_neg:
                        neg.append(node)
                        #newG.nodes[node]['status'] = -1
                        #print(newG.nodes.data('status'))
                for node in pos:
                    newG.nodes[node]['status'] = 1
                    newG.nodes[node]['color'] = 'red'
                for node in neg:
                    newG.nodes[node]['status'] = -1     
                    newG.nodes[node]['color'] = 'blue'
                copy_G = newG.copy()
                self.LTM.append(copy_G)
population = 100
X = HPV_network(population, 7, None)
X.generate_BAGraph()
data_neg = [np.random.randint(24,36) for i in range(4)]
data_neu = [np.random.randint(12,24) for i in range(51)]
data_pos = [np.random.randint(0,12) for i in range(45)]
atttudes = data_neg+data_neu+data_pos
random.shuffle(atttudes)
data = { 'initial attitude': atttudes}
df = pd.DataFrame(data=data)
X.add_attributes_to_nodes(df)
X.add_colors()
X.generate_di_graph(rand=1)
X.run_linear_threshold_model(rand = 1, threshold_pos=4,threshold_neg=-0.45,inital_pos=[],inital_neg=[],num_initial_pos=1,num_initial_neg=1,time_periods=5)
#print(X.G.nodes.data())
t=0
for Gs in X.LTM:
    print('********************')
    print('time',t)
    t+=1
    #print(Gs.nodes.data())
    ls = np.array([Gs.nodes.data('status')[i] for i in Gs.nodes])
    means = np.mean(ls)
    mask_pos = np.where(ls==1)
    print(means)
    age = np.array([Gs.nodes.data('initial attitude')[i] for i in Gs.nodes])
    pos_age = age[mask_pos]
    meansage = np.mean(pos_age)
    print('Num pos',len(mask_pos[0]))
    print('Mean initial att',meansage)
    print('Node 0 Att',Gs.nodes.data('status')[0])
    node_colors = [Gs.nodes.data('color')[i] for i in range(population)]
    nx.draw_networkx(Gs, with_labels=True, node_color=node_colors)
    plt.show()
#print(X.G.edges.data())

def calculate_assortativity(graph, attribute):
    return nx.attribute_assortativity_coefficient(graph, attribute)

assortativity = calculate_assortativity(Gs, 'age')
print("Assortativity:", assortativity)
