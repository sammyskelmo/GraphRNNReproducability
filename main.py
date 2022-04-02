import networkx as nx
import random
import numpy as np
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
********************************
1. Making Graphs
********************************e

Start with the grid generation (as specified by paper)
Max |V| = 361
Max |E| = 684
100 ≤ |V| ≤ 400 implies 10 ≤ side edge ≤ 20
But the test data table implies max node of 19, so we perform the following:
'''


all_G = []
for r in range(10,20):
    for c in range(10,20):
        all_G.append(nx.grid_2d_graph(r,c))

random.seed(420)
random.shuffle(all_G)

#Paper: 80% train, 20% test
te_tr_bound = int(0.8 * len(all_G))
tr_val_bound = int(0.2 * len(all_G))

#Test, Train, Validate
te_G = all_G[te_tr_bound:]
tr_G = all_G[0:te_tr_bound]
val_G = all_G[0:tr_val_bound]

#work out node, edge boundaries
edge_max = max([all_G[g].number_of_edges() for g in range(len(all_G))])
max_nodes = max([all_G[g].number_of_nodes() for g in range(len(all_G))])
edge_min = min([all_G[g].number_of_edges() for g in range(len(all_G))])
