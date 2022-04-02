import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a deterministic linear output
class MLP_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_plain, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        y = self.deterministic_output(h)
        return y

# a deterministic linear output (update: add noise)
class MLP_VAE_conditional_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_conditional_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size)  # lsgms

        self.decode_1 = nn.Linear(embedding_size+h_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size)  # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.shape)).to(device)
        z = eps * z_sgm + z_mu
        # decoder
        y = self.decode_1(torch.cat((h,z),dim=-1))
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms

if __name__ == "__main__":
    print("device:", device)
    _mlp_plain = MLP_plain(16, 32, 8).to(device)
    _input = torch.randn(16).to(device)
    _output = _mlp_plain(_input)
    print(_output.shape)

    _mlp_vae_cond_plain = MLP_VAE_conditional_plain(16, 32, 8).to(device)
    _input = Variable(torch.randn(64, 16)).to(device)
    _y, _z_mu, _z_lsgms = _mlp_vae_cond_plain(_input)
    print(_y.shape, _z_mu.shape, _z_lsgms.shape)