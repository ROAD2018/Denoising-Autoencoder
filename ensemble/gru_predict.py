import torch
from model_building import n_features,hidden_dim,batch_size,seq_len,device,SOS_token
from model_building.grumodel import Encoder,Decoder
from ensemble import path_gru


def get_encoder_decoder_from_gru():
    """
    :return: load gru model from disk
    """
    encoder = Encoder(n_features, hidden_dim,batch_size).to(device)
    decoder = Decoder(seq_len,hidden_dim, n_features,batch_size).to(device)
    checkpoint = torch.load(path_gru)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return encoder, decoder


def predict(input_tensor,encoder,decoder):
    """
    :param input_tensor: input array of len = seq_len
    :param encoder: trained gru encoder model
    :param decoder: trained gru decoder model
    :return: predicted outputs
    """
    input_tensor = input_tensor.view(-1, batch_size)
    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(input_length, batch_size, encoder.hidden_dim,dtype=torch.float32, device=device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] += encoder_output[:, 0]

    decoder_input = torch.tensor([[SOS_token]] * batch_size, dtype=torch.float32, device=device)  # SOS

    decoder_hidden = encoder_hidden
    decoder_ops = torch.zeros(input_length, batch_size, 1, dtype=torch.float32, device=device)
    for di in range(input_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs[-1, :, :])
        decoder_ops[di] = decoder_output.view(batch_size, 1)
        decoder_input = decoder_output.view(batch_size, 1).squeeze().detach()
    return decoder_ops