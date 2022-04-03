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

device = "cuda" if torch.cuda.is_available() else "cpu"


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


# class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
#     def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
#         self.adj_all = []
#         self.len_all = []
#         for G in G_list:
#             self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
#             self.len_all.append(G.number_of_nodes())
#         if max_num_node is None:
#             self.n = max(self.len_all)
#         else:
#             self.n = max_num_node
#         if max_prev_node is None:
#             print('calculating max previous node, total iteration: {}'.format(iteration))
#             self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
#             print('max previous node: {}'.format(self.max_prev_node))
#         else:
#             self.max_prev_node = max_prev_node

#         # self.max_prev_node = max_prev_node

#         # # sort Graph in descending order
#         # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
#         # self.len_all = [self.len_all[i] for i in len_batch_order]
#         # self.adj_all = [self.adj_all[i] for i in len_batch_order]
#     def __len__(self):
#         return len(self.adj_all)

#     def __getitem__(self, idx):
#         adj_copy = self.adj_all[idx].copy()
#         # here zeros are padded for small graph
#         x_batch = np.zeros((self.n, self.max_prev_node))
#         x_batch[0, :] = 1  # the first input token is all ones
#         # here zeros are padded for small graph
#         y_batch = np.zeros((self.n, self.max_prev_node))
#         # generate input x, y pairs
#         len_batch = adj_copy.shape[0]
#         x_idx = np.random.permutation(adj_copy.shape[0])
#         adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
#         adj_copy_matrix = np.asmatrix(adj_copy)
#         G = nx.from_numpy_matrix(adj_copy_matrix)
#         # then do bfs in the permuted G
#         start_idx = np.random.randint(adj_copy.shape[0])
#         x_idx = np.array(bfs(G, start_idx))
#         adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
#         adj_encoded = encode_adj(
#             adj_copy.copy(), max_prev_node=self.max_prev_node)
#         # get x and y and adj
#         # for small graph the rest are zero padded
#         y_batch[0:adj_encoded.shape[0], :] = adj_encoded
#         x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
#         return {'x': x_batch, 'y': y_batch, 'len': len_batch}

#     def calc_max_prev_node(self, iter=20000, topk=10):
#         max_prev_node = []
#         for i in range(iter):
#             if i % (iter / 5) == 0:
#                 print('iter {} times'.format(i))
#             adj_idx = np.random.randint(len(self.adj_all))
#             adj_copy = self.adj_all[adj_idx].copy()
#             # print('Graph size', adj_copy.shape[0])
#             x_idx = np.random.permutation(adj_copy.shape[0])
#             adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
#             adj_copy_matrix = np.asmatrix(adj_copy)
#             G = nx.from_numpy_matrix(adj_copy_matrix)
#             # then do bfs in the permuted G
#             start_idx = np.random.randint(adj_copy.shape[0])
#             x_idx = np.array(bfs(G, start_idx))
#             adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
#             # encode adj
#             adj_encoded = encode_adj_flexible(adj_copy.copy())
#             max_encoded_len = max([len(adj_encoded[i])
#                                    for i in range(len(adj_encoded))])
#             max_prev_node.append(max_encoded_len)
#         max_prev_node = sorted(max_prev_node)[-1*topk:]
#         return max_prev_node


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
    y = F.sigmoid(y)
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
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
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
        h = rnn(x)
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
        y_pred = output(output_x, output_y_len)  # number of arguments error
        # y_pred = output(output_x) #mismatched dimension
        y_pred = F.sigmoid(y_pred)
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
                epoch, args.epochs, loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.data[0]*feature_dim
    return loss_sum/(batch_idx+1)


def train_rnn_epoch_original(epoch, args, rnn, output, data_loader,
                             optimizer_rnn, optimizer_output,
                             scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        print(data['len'])
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        output_x = torch.cat(
            (torch.ones(y_reshape.size(0), 1, 1), y_reshape[:, 0:-1, 0:1]), dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1, 0, -1):
            # count how many y_len is above i
            count_temp = np.sum(output_y_len_bin[i:])
            # put them in output_y_len; max value should not exceed y.size(2)
            output_y_len.extend([min(i, y.size(2))]*count_temp)
        # pack into variable
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        # get packed hidden vector
        h = pack_padded_sequence(h, y_len, batch_first=True).data
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(
            args.num_layers-1, h.size(0), h.size(1))).cuda()
        # num_layers, batch_size, hidden_size
        output.hidden = torch.cat(
            (h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
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
                epoch, args.epochs, loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname,
                  loss.data[0], epoch*args.batch_ratio+batch_idx)
        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.data[0]*feature_dim
    return loss_sum/(batch_idx+1)


def test_train_MLP_Jasper():
    args = Args()

    # official parameter for ENZYMES
    graphs = load_graph_dataset(min_num_nodes=10, name='ENZYMES')
    args.max_prev_node = 25

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
                                             max_num_node=args.max_num_node)

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

        '''
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    G_pred.extend(G_pred_step)
                # save graphs
                # fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                # save_graph_list(G_pred, fname)

            print('test done, graphs saved')

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        '''

        epoch += 1

    # np.save(args.timing_save_path+args.fname,time_all)


def test_train_rnn_Penny():
    args = Args()
    # graphs = load_graph_dataset(min_num_nodes=10, name='PROTEINS_full')
    # args.max_prev_node = 230  # M = 230 is suggested by the paper for protein graph

    # official parameter for ENZYMES
    graphs = load_graph_dataset(min_num_nodes=10, name='ENZYMES')
    args.max_prev_node = 3
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
                                             max_num_node=args.max_num_node)

    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size * args.batch_ratio,
                                                                     replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 sampler=sample_strategy)

    rnn = GRU_base(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                   hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, prepend_linear_layer=True,
                   append_linear_layers=False).to(device)
    output = GRU_base(input_size=1, embedding_size=args.embedding_size_rnn_output,
                      hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers).to(device)

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
        epoch += 1


if __name__ == '__main__':
    test_train_rnn_Penny()
