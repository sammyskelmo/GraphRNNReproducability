import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
import random
from random import shuffle
import pickle
import scipy.misc
import time as tm
from IPython import embed
import os
import argparse

# from utils import *
# from model import *
# from data import *
# import create_graphs

from GRU_base import *
from LSTM_base import *
from MLP_base import *
from load_datasets import *
from args import Args
from args import Args
# from model import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def manual_train_args():
    parser = argparse.ArgumentParser(description="manual_train_args")
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--graph_type", type=str, default=None)
    parser.add_argument("--bfs_mode", type=str, default=None)

    manual_args = parser.parse_args()
    return manual_args

def create_save_path(_arg):
    if not os.path.exists(_arg.model_save_path):
        os.mkdir(_arg.model_save_path)
    if not os.path.exists(_arg.graph_save_path):
        os.mkdir(_arg.graph_save_path)
    if not os.path.exists(_arg.figure_save_path):
        os.mkdir(_arg.figure_save_path)

def encode_adj(adj, max_prev_node=10, is_full=False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]  # reverse order

    return adj_output


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full

def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G

# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)

def binary_cross_entropy_weight(y_pred, y, has_weight=False, weight_length=1, weight_max=10):
    '''

    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    '''
    if has_weight:
        weight = torch.ones(y.size(0), y.size(1), y.size(2))
        weight_linear = torch.arange(
            1, weight_length+1)/weight_length*weight_max
        weight_linear = weight_linear.view(
            1, weight_length, 1).repeat(y.size(0), 1, y.size(2))
        weight[:, -1*weight_length:, :] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight.to(device))
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float().to(device)
        y_unsorted = data['y'].float().to(device)
        y_len_unsorted = data['len'].to(device)
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :].to(device)
        y_unsorted = y_unsorted[:, 0:y_len_max, :].to(device)
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        sort_index.to(device)
        y_len = y_len.cpu().numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = Variable(x).to(device)
        y = Variable(y).to(device)

        h = rnn(x)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        if epoch % args.epochs_log == 0 and batch_idx == 0:  # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss.data, args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        # log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.data
    return loss_sum/(batch_idx+1)


def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y = torch.sigmoid(y)
    # do sampling
    if sample:
        if sample_time > 1:
            y_result = Variable(torch.rand(
                y.size(0), y.size(1), y.size(2))).to(device)
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = Variable(torch.rand(
                        y.size(1), y.size(2))).to(device)
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(
                y.size(0), y.size(1), y.size(2))).to(device)
            y_result = torch.gt(y, y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(
            y.size(0), y.size(1), y.size(2))*thresh).to(device)
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(
        device)  # normalized prediction score
    y_pred_long = Variable(torch.zeros(
        test_batch_size, max_num_node, args.max_prev_node)).to(device)  # discrete prediction
    x_step = Variable(torch.ones(test_batch_size, 1,
                                 args.max_prev_node)).to(device)
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True,
                                sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(device)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list


def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        data = reformat_data(data)
        rnn.zero_grad()
        output.zero_grad()
        x = data['x']
        y = data['y']
        y_len = data['y_len']
        output_x = data['output_x']
        output_y = data['output_y']
        output_y_len = data['output_y_len']

        x = x.to(device)
        y = y.to(device)
        output_x = output_x.to(device)
        output_y = output_y.to(device)
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())
        rnn.hidden = rnn.init_hidden(batch_size=x.size(0))

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)

        # get packed hidden vector
        h = pack_padded_sequence(h, y_len, batch_first=True).data
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(device)
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(
            args.num_layers-1, h.size(0), h.size(1))).to(device)
        # num_layers, batch_size, hidden_size
        output.hidden = torch.cat(
            (h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)

        y_pred = output(output_x, pack=True, input_len=output_y_len)

        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(
            output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        if epoch % args.epochs_log == 0 and batch_idx == 0:  # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss.data, args.graph_type, args.num_layers, args.hidden_size_rnn))

        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.data*feature_dim
    return loss_sum/(batch_idx+1)

def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).cuda()
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).cuda()
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


def test_train_MLP_Jasper(args):
    args = reset_args(args)
    create_save_path(args)

    if args.graph_type == 'grid':
        graphs = []
        for i in range(10,20):
            for j in range(10,20):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 40
    elif args.graph_type == 'enzymes':
        graphs = load_graph_dataset(min_num_nodes=10, name='ENZYMES')
        args.max_prev_node = 25
    elif args.graph_type=='barabasi':
        graphs = []
        for i in range(100,200):
             for j in range(4,5):
                 for k in range(5):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 130
    elif args.graph_type == 'DD':
        graphs = load_graph_dataset(min_num_nodes=100, max_num_nodes=500, name='DD',node_attributes=False,graph_labels=True)
        args.max_prev_node = 230
    elif args.graph_type == 'citeseer':
        _, _, G = Graph_load(dataset='citeseer')
        G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)
        args.max_prev_node = 250
    elif args.graph_type == 'citeseer_small':
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
        args.max_prev_node = 15
    elif args.graph_type.startswith('community'):
        num_communities = int(args.graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')
        # c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        # c_sizes = [15] * num_communities
        graphs = []
        for k in range(500):
            c_sizes = list(np.random.choice(range(30, 81), 1)) * num_communities
            graphs.append(n_community(c_sizes, p_inter=0.05))
        args.max_prev_node = 80
    else:
        print('no dataset')


    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8 * graphs_len)]
    graphs_validate = graphs[0:int(0.2 * graphs_len)]

    args.max_num_node = max([graphs[i].number_of_nodes()
                             for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges()
                        for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges()
                        for i in range(len(graphs))])

    # args.max_num_node = 2000
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(
        len(graphs), len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    dataset = Graph_sequence_sampler_pytorch(graphs_train, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node, bfs_mode=args.bfs_mode)

    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size * args.batch_ratio,
                                                                     replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 sampler=sample_strategy)

    rnn = GRU_base(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                   hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, prepend_linear_layer=True,
                   append_linear_layers=False).to(device)
    output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output,
                       y_size=args.max_prev_node).to(device)

    epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(
        optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(
        optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train

        train_mlp_epoch(epoch, args, rnn, output, dataset_loader,
                        optimizer_rnn, optimizer_output,
                        scheduler_rnn, scheduler_output)

        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start

        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break

            print('test done, graphs saved')

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)


        epoch += 1

    # np.save(args.timing_save_path+args.fname,time_all)


def test_train_rnn_Penny(args):
    args = reset_args(args)
    create_save_path(args)
    # graphs = load_graph_dataset(min_num_nodes=10, name='PROTEINS_full')
    # args.max_prev_node = 230  # M = 230 is suggested by the paper for protein graph

    # official parameter for ENZYMES
    if args.graph_type == 'grid':
        graphs = []
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 40
    elif args.graph_type == 'enzymes':
        graphs = load_graph_dataset(min_num_nodes=10, name='ENZYMES')
        args.max_prev_node = 25
    elif args.graph_type == 'barabasi':
        graphs = []
        for i in range(100, 200):
            for j in range(4, 5):
                for k in range(5):
                    graphs.append(nx.barabasi_albert_graph(i, j))
        args.max_prev_node = 130
    elif args.graph_type == 'DD':
        graphs = load_graph_dataset(min_num_nodes=100, max_num_nodes=500, name='DD', node_attributes=False,
                                    graph_labels=True)
        args.max_prev_node = 230
    elif args.graph_type == 'citeseer':
        _, _, G = Graph_load(dataset='citeseer')
        G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)
        args.max_prev_node = 250
    elif args.graph_type == 'citeseer_small':
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
        args.max_prev_node = 15
    elif args.graph_type.startswith('community'):
        num_communities = int(args.graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')
        # c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        # c_sizes = [15] * num_communities
        graphs = []
        for k in range(500):
            c_sizes = list(np.random.choice(range(30, 81), 1)) * num_communities
            graphs.append(n_community(c_sizes, p_inter=0.05))
        args.max_prev_node = 80
    else:
        print('no dataset')

    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8 * graphs_len)]
    graphs_validate = graphs[0:int(0.2 * graphs_len)]

    args.max_num_node = max([graphs[i].number_of_nodes()
                             for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges()
                        for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges()
                        for i in range(len(graphs))])

    print('total graph num: {}, training set: {}'.format(
        len(graphs), len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    print('max previous node: {}'.format(args.max_prev_node))

    dataset = Graph_sequence_sampler_pytorch(graphs_train, max_prev_node=args.max_prev_node,
                                             max_num_node=args.max_num_node, bfs_mode=args.bfs_mode)

    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size * args.batch_ratio,
                                                                     replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 sampler=sample_strategy)

    rnn = GRU_base(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, prepend_linear_layer=True,
                        append_linear_layers=True, append_linear_output_size=args.hidden_size_rnn_output).to(device)
    output = GRU_base(input_size=1, embedding_size=args.embedding_size_rnn_output,
                           hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, prepend_linear_layer=True,
                           append_linear_layers=True, append_linear_output_size=1).to(device)

    epoch = 1

    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(
        optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(
        optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        train_rnn_epoch(epoch, args, rnn, output, dataset_loader,
                        optimizer_rnn, optimizer_output,
                        scheduler_rnn, scheduler_output)

        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start

        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                while len(G_pred) < args.test_total_size:
                    G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) + '_' + str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break

            print('test done, graphs saved')

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)

        epoch += 1

def reset_args(_args):
    _args.fname = _args.note + '_' + _args.graph_type + '_' + str(_args.num_layers) + '_' + str(_args.hidden_size_rnn) + '_'
    _args.fname_pred = _args.note + '_' + _args.graph_type + '_' + str(_args.num_layers) + '_' + str(
        _args.hidden_size_rnn) + '_pred_'
    _args.fname_train = _args.note + '_' + _args.graph_type + '_' + str(_args.num_layers) + '_' + str(
        _args.hidden_size_rnn) + '_train_'
    _args.fname_test = _args.note + '_' + _args.graph_type + '_' + str(_args.num_layers) + '_' + str(
        _args.hidden_size_rnn) + '_test_'
    _args.fname_baseline = _args.graph_save_path + _args.graph_type + _args.generator_baseline + '_' + _args.metric_baseline
    return _args

if __name__ == '__main__':
    args = Args()
    manual_args = manual_train_args()
    if manual_args.note is not None:
        args.note = manual_args.note
    if manual_args.graph_type is not None:
        args.graph_type = manual_args.graph_type
    if manual_args.bfs_mode is not None:
        args.bfs_mode = manual_args.bfs_mode
    print('note:', args.note, 'graph_type:', args.graph_type, 'bfs_mode:', args.bfs_mode)
    if manual_args.graph_type == 'loop':
        for g_type in ['community4', 'grid', 'citeseer', 'DD']:
            args.graph_type = g_type
            print("{:-^60s}".format('LOOP'))
            print('note:', args.note, 'graph_type:', args.graph_type)
            if 'GraphRNN_MLP' in args.note:
                test_train_MLP_Jasper(args)
            elif 'GraphRNN_RNN' in args.note:
                test_train_rnn_Penny(args)
            else:
                print('note no found')
    else:
        if 'GraphRNN_MLP' in args.note:
            test_train_MLP_Jasper(args)
        elif 'GraphRNN_RNN' in args.note:
            test_train_rnn_Penny(args)
        else:
            print('note no found')

