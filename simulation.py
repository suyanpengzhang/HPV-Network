#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:03:21 2023

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
import pickle
warnings.filterwarnings('ignore')
# =============================================================================
# SECTION 5: HPV VACCINE SOCIAL NORMS
# 59.Have you ever talked about cervical cancer or HPV vaccine with your daughter?
# 60.How comfortable do you feel talking about cervical cancer or HPV vaccine with your daughter?
# 61.Have you ever talked about cervical cancer or HPV vaccine with other parents?
# 62.How comfortable do you feel talking about cervical cancer or HPV vaccine with other parents?
# 63.I feel that other parents in my community are vaccinating their children with routine childhood vaccines
# 64.I feel that other parents in my community are vaccinating their daughters against HPV
# 65.Is there anyone you talk with regularly who had the opportunity to vaccinate their daughter against HPV but declined to do so?
# 66.Have you ever talked about cervical cancer or HPV vaccine with a doctor / nurse / other health care worker?
# larger --> more talktive
# =============================================================================

# =============================================================================
# sec 6
# How much do you trust each of the following in general?
# 67............the people in your community
# 68.................the national government
# 69.… the county government
# 70… doctors and nurses
# 71… community health workers/volunteers
# 72… people who work at non-governmental organizations/civil society
# 73… traditional healers
# 74… religious leaders
# smaller -> more trust
# =============================================================================

# =============================================================================
# Sec 8
# 89.I am always courteous even to people who are disagreeable
# 90.There have been occasions when I took advantage of someone.
# 91.I sometimes try to “get even” rather than “forgive and forget.”
# 92.I sometimes feel resentful when I don’t get my way.
# 93.No matter who I am talking to, I’m always a good listener
# 89,93 and 91,92 opposite
# =============================================================================

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
#hpvdata = hpvdata.sample(30)
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

Network = HPVN.HPV_network(num_household, 7, None) #7 is num edge attached
Network.generate_BAGraph()
#attitides
id_ = [i for i in hpvdata.index]
attitudes = [hpvdata['HPV_VAX_attitu_s35'][i] for i in hpvdata.index]
# =============================================================================
# stub1 = [hpvdata['sec8_q91'][i] if hpvdata['sec8_q91'][i]>0 else 2.5 for i in hpvdata.index]
# =============================================================================
stub2 = [(hpvdata['sec8_q92'][i]-min_stub)/(max_stub-min_stub) if hpvdata['sec8_q92'][i]>0 else (np.nanmean(hpvdata[hpvdata['sec8_q92']>0]['sec8_q92'])-min_stub)/(max_stub-min_stub) for i in hpvdata.index]
# =============================================================================
# listen1 = [hpvdata['sec8_q89'][i] if hpvdata['sec8_q89'][i]>0 else 2.5 for i in hpvdata.index]
# listen2 = [hpvdata['sec8_q93'][i] if hpvdata['sec8_q93'][i]>0 else 2.5 for i in hpvdata.index]
# =============================================================================
good_listen_score =[stub2[i]for i in range(len(stub2))]
# the larger the number is, less stubborn
data = { 'id':id_, 'initial attitude': attitudes,'current attitude': attitudes, 'listen_score':good_listen_score}
df = pd.DataFrame(data=data)
Network.add_attributes_to_nodes(df)
Network.add_colors([12,24])
Network.generate_di_graph(social_connectivity)
Network.normalize_edge_weights()
#pickle.dump(Network, open("simple_net.pkl", "wb"))
#Network.run_linear_threshold_model_soft(inital_threshold=[12,24],time_periods=20)
for _lambda in np.arange(0,1.1,0.1):
    print(_lambda)
    data = np.zeros((len(np.arange(0,10.5,0.5)),len(np.arange(-5,0.5,0.5))))
    count_x = 0
    for pos_thre in np.arange(0,10.5,0.5):
        count_y = 0
        for neg_thre in np.arange(-5,0.5,0.5):
            Network.run_linear_threshold_model(lambda_ = _lambda,threshold_pos=pos_thre,threshold_neg=neg_thre,inital_threshold=[12,24],time_periods=5)
            Gs = Network.LTM[-1]
            ls = np.array([Gs.nodes.data('status')[i] for i in Gs.nodes])
            mask_pos = np.where(ls==1)[0]
            mask_neg = np.where(ls==-1)[0]
            data[count_x,count_y] = len(mask_pos)
            count_y += 1
        count_x += 1
    fig = plt.figure(figsize = (6, 4),dpi=300)
    plt.title('lambda = '+str(_lambda)+', Number of positive attitudes')
    ax = sns.heatmap(data)
    plt.xlabel("neg_threshold")
    plt.xticks(np.arange(0,11,2),np.arange(-5,0.5,1))
    plt.ylabel("pos_threshold")
    plt.yticks(np.arange(0,21,2),np.arange(0,10.5,1))
    ax.invert_yaxis()
    plt.show()


edge_weights =[]
for edge in Network.G.edges:
    if Network.G.edges[edge]['weight']<0:
        print(edge)
    edge_weights.append(Network.G.edges[edge]['weight'])
fig = plt.figure(figsize = (6, 4),dpi=600)
# creating the bar plot
plt.hist(good_listen_score,bins=4, color ='maroon')
plt.xlabel("score")
plt.show()


fig = plt.figure(figsize = (6, 4),dpi=600)

plt.hist(edge_weights,bins=10, color ='maroon')
 
plt.xlabel("score")
plt.show()
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
