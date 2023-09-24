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
# higher the score, more negative
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
    trusti = np.nanmean(np.array([hpvdata['sec6_q67'][i],hpvdata['sec6_q68'][i],
                                 hpvdata['sec6_q69'][i],hpvdata['sec6_q70'][i],
                                 hpvdata['sec6_q71'][i],hpvdata['sec6_q72'][i],
                                 hpvdata['sec6_q73'][i],hpvdata['sec6_q74'][i]]))
    trustj = np.nanmean(np.array([hpvdata['sec6_q67'][j],hpvdata['sec6_q68'][j],
                                 hpvdata['sec6_q69'][j],hpvdata['sec6_q70'][j],
                                 hpvdata['sec6_q71'][j],hpvdata['sec6_q72'][j],
                                 hpvdata['sec6_q73'][j],hpvdata['sec6_q74'][j]]))
    listeni = np.nanmean(np.array([hpvdata['sec8_q89'][i],hpvdata['sec8_q93'][i]]))
    listenj = np.nanmean(np.array([hpvdata['sec8_q89'][j],hpvdata['sec8_q93'][j]]))
    stubborni = np.nanmean(np.array([hpvdata['sec8_q91'][i],hpvdata['sec8_q92'][i]]))
    stubbornj = np.nanmean(np.array([hpvdata['sec8_q91'][j],hpvdata['sec8_q92'][j]]))
    talki = np.nanmean(np.array([hpvdata['sec5_q59'][i],hpvdata['sec5_q60'][i],
                                 hpvdata['sec5_q61'][i],hpvdata['sec5_q62'][i],
                                 hpvdata['sec5_q63'][i],hpvdata['sec5_q64'][i],
                                 hpvdata['sec5_q65'][i],hpvdata['sec5_q66'][i]]))
    talkj = np.nanmean(np.array([hpvdata['sec5_q59'][j],hpvdata['sec5_q60'][j],
                                 hpvdata['sec5_q61'][j],hpvdata['sec5_q62'][j],
                                 hpvdata['sec5_q63'][j],hpvdata['sec5_q64'][j],
                                 hpvdata['sec5_q65'][j],hpvdata['sec5_q66'][j]]))
    if ((talki)*(1/trustj)*(1/listenj)*stubbornj>0) or ((talki)*(1/trustj)*(1/listenj)*stubbornj<0):
        return (talki)*(1/trustj)*(1/listenj)*stubbornj
    else:
        return 0

##read data
file_path = 'hpvdata.csv'
hpvdata = pd.read_csv(file_path)
hpvdata = hpvdata.dropna(subset=['HPV_VAX_attitu_s35'])

num_household = len(hpvdata)

Network = HPVN.HPV_network(num_household, 7, None) #7 is num edge attached
Network.generate_BAGraph()
#attitides
id_ = [i for i in hpvdata.index]
attitudes = [hpvdata['HPV_VAX_attitu_s35'][i] for i in hpvdata.index]
data = { 'id':id_, 'initial attitude': attitudes}
df = pd.DataFrame(data=data)
Network.add_attributes_to_nodes(df)
Network.add_colors([12,24])
Network.generate_di_graph(social_connectivity)
Network.run_linear_threshold_model(rand = 1, threshold_pos=10,threshold_neg=-10,inital_threshold=[12,24],time_periods=5)

t=0
for Gs in Network.LTM:
    print('********************')
    print('time',t)
    t+=1
    ls = np.array([Gs.nodes.data('status')[i] for i in Gs.nodes])
    means = np.mean(ls)
    mask_pos = np.where(ls==1)
    mask_neg = np.where(ls==-1)
    print(means)
    age = np.array([Gs.nodes.data('initial attitude')[i] for i in Gs.nodes])
    pos_age = age[mask_pos]
    meansage = np.mean(pos_age)
    print('Num pos',len(mask_pos[0]))
    print('Num negative',len(mask_neg[0]))
    print('Mean initial att among pos',meansage)
    #node_colors = [Gs.nodes.data('color')[i] for i in range(num_household)]
    #nx.draw_networkx(Gs, with_labels=True, node_color=node_colors)
    plt.show()