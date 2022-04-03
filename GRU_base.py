import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class GRU_base(nn.Module):
    '''
    A Gated Recurrent Unit
    '''

    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, num_layers: int, prepend_linear_layer: bool = True, append_linear_layers: bool = False, append_linear_output_size: int = None):
        '''
            params:
            input_size: the dimension of the input to this class
            embedding_size: the dimension of the embedding vector (different from input_size if prepend_linear_layer is True)
            hidden_size: the dimension of the hidden state
            num_layers: the number of GRU layers
            prepend_linear_layer: add a linear layer before the GRU
            append_linear_layers: add a linear layer after the GRU
            append_linear_output_size: the dimension of the output of the linear layer after the GRU
        '''

        super(GRU_base, self).__init__()

        # save all args for later use
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prepend_linear_layer = prepend_linear_layer
        self.append_linear_layers = append_linear_layers
        self.append_linear_output_size = append_linear_output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.hidden = None

        if prepend_linear_layer:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        if append_linear_layers:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, append_linear_output_size)
            )

        self.init_params()

    def init_params(self):
        # in the rnn, xavier for weights, constant 0.25 for biases
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
        # in the linear layers, 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # could try with and without gain (paper uses gain)
                m.weight.data = init.xavier_uniform_(m.weight.data)
                # m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        ''' caution: the output tensor needs to be sent to the right device '''
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device)

    ### TODO : add pack_padded_sequence, understand what packing is for and does
    def forward(self, base_input):
        if self.prepend_linear_layer:
            input = F.relu(self.input(base_input))
        else:
            input = base_input

        output, self.hidden = self.rnn(input, self.hidden)
        
        if self.append_linear_layers:
            output = self.output(output)

        return output

