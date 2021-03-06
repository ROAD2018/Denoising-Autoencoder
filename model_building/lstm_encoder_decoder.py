import random
from torch import nn
from model_building import *


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        """
        :param encoder: input lstm/gru encoder
        :param decoder: input lstm/gru decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,input_tensor, target_tensor):
        target_length = target_tensor.size(0)
        batchsize = input_tensor.size(1)
        encoder_hidden,encoder_cell = self.encoder(input_tensor)
        decoder_ops = torch.zeros(target_length, batchsize, 1, device=device)
        decoder_input = torch.tensor([[SOS_token]] * batchsize, device=device)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        for di in range(target_length):
            decoder_output, decoder_hidden,decoder_cell = self.decoder(
                decoder_input, decoder_hidden, decoder_cell)
            decoder_ops[di] = decoder_output.view(batchsize, 1)
            decoder_input = target_tensor[di] if use_teacher_forcing else decoder_output.view(batchsize, 1).squeeze().detach()
        return decoder_ops


class Encoder(nn.Module):
    def __init__(self, n_features,seq_len =24, hidden_dim=64, batch_size=32, num_layers=1):
        """
        :param n_features: int - num of features for each input (default = 1)
        :param seq_len: int - seq_length of encoder (number of cells in the seq model at the encoder stage)
        :param hidden_dim: int - dimensions of the hidden layer
        :param batch_size: int - batch size of the input data
        :param num_layers: int - number of layers in the LSTM model (default =1)
        """
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x):
        x = x.reshape((self.batch_size, self.seq_len, -1))
        x, (hidden_n,cell_n) = self.rnn1(x)
        return hidden_n,cell_n


class Decoder(nn.Module):
    def __init__(self, seq_len, hidden_dim=64, n_features=1, batch_size=32, num_layers=1):
        """
         :param seq_len: int - seq_length of decoder (number of cells in the seq model at the decoder)
        :param hidden_dim: int - dimensions of the hidden layer
        :param n_features: int - num of features for each input (default = 1)
        :param batch_size: int - batch size of the input data
        :param num_layers: int - number of layers in the LSTM model (default =1)
        """
        super(Decoder, self).__init__()
        self.input_dim = hidden_dim
        self.hidden_dim, self.n_features = hidden_dim, n_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, hidden_n, cell_n):
        x = x.view(self.batch_size,1, -1)
        output, (hidden_n,cell_n) = self.rnn1(x, (hidden_n,cell_n))
        return self.output_layer(output), hidden_n,cell_n
