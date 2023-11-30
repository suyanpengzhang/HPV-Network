#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:01:16 2023

@author: suyanpengzhang
"""

import glfw
import time

from OpenGL.GL import *
import numpy as np
import random
import math
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

with open("network_example/simple_net.pkl", "rb") as file:
    Network = pickle.load(file)

Network.run_linear_threshold_model(lambda_ = 0.6,threshold_pos=2,threshold_neg=-0.3,inital_threshold=[12,20],time_periods=10)


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

# Let's assume you have three networkx graphs already created: G1, G2, G3
# For example:
G1 = Network.LTM[0]
G2 = Network.LTM[1]
G3 = Network.LTM[2]
G4 = Network.LTM[3]
G5 = Network.LTM[4]
G6 = Network.LTM[5]
G7 = Network.LTM[6]
G8 = Network.LTM[7]
G9 = Network.LTM[8]
G10 = Network.LTM[9]

# Function to extract data from networkx graph
def extract_graph_data(G):
    positions = nx.spring_layout(G)  # or any other layout
    node_statuses = G.nodes.data('status')
    edge_weights = nx.get_edge_attributes(G, 'weight')
    return positions, node_statuses, edge_weights

# Extract data from each graph
positions1, node_statuses1, edge_weights1 = extract_graph_data(G1)
positions2, node_statuses2, edge_weights2 = extract_graph_data(G2)
positions3, node_statuses3, edge_weights3 = extract_graph_data(G3)
positions4, node_statuses4, edge_weights4 = extract_graph_data(G4)
positions5, node_statuses5, edge_weights5 = extract_graph_data(G5)
positions6, node_statuses6, edge_weights6 = extract_graph_data(G6)
positions7, node_statuses7, edge_weights7 = extract_graph_data(G7)
positions8, node_statuses8, edge_weights8 = extract_graph_data(G8)
positions9, node_statuses9, edge_weights9 = extract_graph_data(G9)
positions10, node_statuses10, edge_weights10 = extract_graph_data(G10)

# Now you can use positions and edge_weights to draw your graphs with OpenGL
# Define the node class
class Node:
    def __init__(self, position,status):
        self.position = position
        self.status = status  # Random initial status

    # Function to draw the node
    def draw(self):
        glColor3f(0.5, 0.5, 0.5) if self.status == 0 else glColor3f(1.0, 0.0, 0.0) if self.status == -1 else glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_POLYGON)
        for i in range(360):
            theta = 2.0 * math.pi * i / 360
            x = 0.05 * math.cos(theta) + self.position[0]
            y = 0.05 * math.sin(theta) + self.position[1]
            glVertex2f(x, y)
        glEnd()

# Function to draw the edges
def draw_edge(start, end, weight):
    glLineWidth(weight * 10)  # Scale the weight for visibility
    glBegin(GL_LINES)
    glVertex2f(start[0], start[1])
    glVertex2f(end[0], end[1])
    glEnd()
# Get the start time
start_time = time.time()
# Set the duration for each graph display
display_duration = 0.5  # seconds

# Initialize GLFW
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# Create a GLFW window
window = glfw.create_window(720, 720, "Network Simulation", None, None)

# Make the context current
glfw.make_context_current(window)

# Create a grid of nodes
network1 = [Node((positions1[i][0],positions1[i][1]),node_statuses1[i]) for i in range(30)]
network2 = [Node((positions1[i][0],positions1[i][1]),node_statuses2[i]) for i in range(30)]
network3 = [Node((positions1[i][0],positions1[i][1]),node_statuses3[i]) for i in range(30)]
network4 = [Node((positions1[i][0],positions1[i][1]),node_statuses4[i]) for i in range(30)]
network5 = [Node((positions1[i][0],positions1[i][1]),node_statuses5[i]) for i in range(30)]
network6 = [Node((positions1[i][0],positions1[i][1]),node_statuses6[i]) for i in range(30)]
network7 = [Node((positions1[i][0],positions1[i][1]),node_statuses7[i]) for i in range(30)]
network8 = [Node((positions1[i][0],positions1[i][1]),node_statuses8[i]) for i in range(30)]
network9 = [Node((positions1[i][0],positions1[i][1]),node_statuses9[i]) for i in range(30)]
network10 = [Node((positions1[i][0],positions1[i][1]),node_statuses10[i]) for i in range(30)]


# Randomly generate weights for edges between nodes (for simplicity, fully connected)
#weights = {(i, j): random.uniform(0.1, 1.0) for i in range(len(network)) for j in range(i+1, len(network))}
weights =edge_weights1
for i in weights:
    weights[i] += 0.01
# Main rendering loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT)

    # Draw the edges
# =============================================================================
#     for (i, j), weight in weights.items():
#         draw_edge(network[i].position, network[j].position, weight)
# =============================================================================
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time < display_duration:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network1:
            node.draw()
    elif elapsed_time < display_duration * 2:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network2:
            node.draw()
    elif elapsed_time < display_duration * 3:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network3:
            node.draw()
    elif elapsed_time < display_duration * 4:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network4:
            node.draw()
    elif elapsed_time < display_duration * 5:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network5:
            node.draw()
    elif elapsed_time < display_duration * 6:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network6:
            node.draw()
    elif elapsed_time < display_duration * 7:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network7:
            node.draw()
    elif elapsed_time < display_duration * 8:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network8:
            node.draw()
    elif elapsed_time < display_duration * 9:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network9:
            node.draw()
    else:
        drawn_edges = set()
        for (i, j), weight in weights.items():
            if (j, i) not in drawn_edges:  # This check avoids drawing the edge twice
                draw_edge(network1[i].position, network1[j].position, weight)
                drawn_edges.add((i, j))
    
        # Draw the network nodes
        for node in network10:
            node.draw()
        # Reset the timer if you want to loop the simulation
        if elapsed_time > display_duration * 10:
            start_time = time.time()

    # Swap front and back buffers
    glfw.swap_buffers(window)

    # Sleep to make the changes visible
    glfw.wait_events_timeout(0.5)  # Update every half second

# Terminate GLFW
glfw.terminate()

