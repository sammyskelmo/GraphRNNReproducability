
import numpy as np
from random import shuffle

# from utils import *
# from model import *
# from data import *
# import create_graphs

from GRU_base import *
from LSTM_base import *
from MLP_base import *
from load_datasets import *
from args import Args
# from model import *

import numbers
import os
import random
import networkx as nx

import eval.stats
import utils
import load_datasets
import model as m

# device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(graph_ref_list, graph_pred_list, num_pred_samples=20):
    graph_pred_list = graph_pred_list[:num_pred_samples]
    # graph_ref_len = len(graph_ref_list)
    random.shuffle(graph_ref_list)
    # test on a hold out test set
    mmd_degree = eval.stats.degree_stats(graph_ref_list, graph_pred_list)
    mmd_clustering = eval.stats.clustering_stats(
        graph_ref_list, graph_pred_list)
    try:
        mmd_4orbits = eval.stats.orbit_stats_all(
            graph_ref_list, graph_pred_list)
    except:
        mmd_4orbits = -1
    print('deg: ', mmd_degree)
    print('clustering: ', mmd_clustering)
    print('orbits: ', mmd_4orbits)


def test_evaluate_model(pred_name, graph_type):
    # pred_name = "graphs/GraphRNN_RNN_grid_4_128_pred_10_1.dat"  # load a model
    pred_graphs = utils.load_graph_list(pred_name)
    if 'grid' in graph_type:
        graphs = []
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))
    elif graph_type == 'enzymes':
        graphs = load_graph_dataset(min_num_nodes=10, name='ENZYMES')
    elif graph_type == 'barabasi':
        graphs = []
        for i in range(100, 200):
            for j in range(4, 5):
                for k in range(5):
                    graphs.append(nx.barabasi_albert_graph(i, j))
    elif graph_type == 'DD':
        graphs = load_graph_dataset(min_num_nodes=100, max_num_nodes=500, name='DD', node_attributes=False,
                                    graph_labels=True)
    elif graph_type == 'citeseer':
        _, _, G = Graph_load(dataset='citeseer')
        G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)
    elif graph_type == 'citeseer_small':
        _, _, G = Graph_load(dataset='citeseer')
        G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        shuffle(graphs)
        graphs = graphs[0:200]
    elif graph_type.startswith('community'):
        num_communities = int(graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')
        c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        # c_sizes = [15] * num_communities
        graphs = []
        for k in range(3000):
            graphs.append(n_community(c_sizes, p_inter=0.01))
    else:
        print('no dataset')
    ref_graphs = graphs
    random.seed(123)
    random.shuffle(ref_graphs)
    ref_graphs_len = len(ref_graphs)
    ref_graphs = ref_graphs[int(0.8 * ref_graphs_len):]
    evaluate_model(ref_graphs, pred_graphs)

if __name__ == '__main__':
    # model_list = []
    note_list = ['GraphRNN_RNN', 'GraphRNN_MLP']
    dataset_list = ['grid', 'community4', 'citeseer', 'DD', 'bfs_min_grid', 'bfs_ran_grid', 'bfs_max_grid', 'bfs_no_grid']
    epoch = 500
    # model_name = "graphs/GraphRNN_RNN_grid_4_128_pred_10_1.dat"
    for _note in note_list:
        for _dataset in dataset_list:
            pred_name = 'graphs/' + _note + '_' + _dataset + '_4_128_pred_' + str(epoch) + '_1.dat'
            if os.path.exists(pred_name):
                print("{:-^60s}".format(pred_name))
                test_evaluate_model(pred_name, _dataset)
            else:
                print("{:*^60s}".format(pred_name + ' not found'))
