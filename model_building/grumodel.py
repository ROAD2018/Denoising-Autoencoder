import copy
import torch
from torch import nn
from model_building import device
RANDOM_SEED = 557


class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64, batch_size=32, num_layers=1):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = embedding_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn1 = nn.GRU(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x, hidden_n):
        x = x.reshape((self.batch_size, 1, -1))
        x, hidden_n = self.rnn1(x, hidden_n)
        #         print("encoder x : {} hidden: {} ".format(x.size(),hidden_n.size()))
        return x, hidden_n

    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=device)


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1, batch_size=32, num_layers=1):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        #         self.attn = nn.Linear(self.input_dim +self.n_features, self.seq_len)
        self.attn_combine = nn.Linear(self.input_dim + self.n_features, self.input_dim)
        self.rnn1 = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, hidden_n, encoder_op):
        x = x.view(1, self.batch_size, -1)
        encoder_op = encoder_op.reshape((self.batch_size, self.input_dim))
        output = torch.cat((x[0], encoder_op.view(1, self.batch_size, self.input_dim)[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output, hidden_n = self.rnn1(output.view(self.batch_size, 1, -1), hidden_n)
        return self.output_layer(output), hidden_n

    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=device)
