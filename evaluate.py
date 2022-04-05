
import numbers
import os
import utils
import random
from torch.autograd import Variable
import torch
import networkx as nx

import eval.stats
import utils
import load_datasets
import model as m

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def test_evaluate_model():
    model_name = "graphs/GraphRNN_RNN_grid_4_128_pred_10_1.dat"  # load a model
    pred_graphs = utils.load_graph_list(model_name)
    ref_graphs = []
    for i in range(10, 20):
        for j in range(10, 20):
            ref_graphs.append(nx.grid_2d_graph(i, j))
    # graphs = load_datasets.load_graph_dataset(min_num_nodes=10, name='ENZYMES')
    # max_prev_node = 3
    random.seed(123)
    random.shuffle(ref_graphs)
    ref_graphs_len = len(ref_graphs)
    ref_graphs = ref_graphs[int(0.8 * ref_graphs_len):]
    evaluate_model(ref_graphs, pred_graphs)


test_evaluate_model()
