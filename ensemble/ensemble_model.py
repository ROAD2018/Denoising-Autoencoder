import torch
from torch import nn
from model_building import seq_len,eps,epochs,batch_size
import ensemble.gru_predict as gru
import ensemble.lstm_predict as lstm
from ensemble import path_ensemble


class Ensemble(nn.Module):
    def __init__(self,lstm_model,gru_encoder,gru_decoder):
        """
        :param lstm_model: trained lstm model on ameriflux soil moisture data to eleiminate noise in the inputs
        :param gru_encoder:  trained gru encoder on ameriflux soil moisture data to eleiminate noise in the inputs
        :param gru_decoder:  trained gru decoder on ameriflux soil moisture data to eleiminate noise in the inputs
        """
        super().__init__()
        self.lstm_model = lstm_model
        self.gru_encoder = gru_encoder
        self.gru_decoder = gru_decoder
        self.fc = nn.Linear(seq_len,seq_len)

    def forward(self,noisy_seq):
        enc, dec = gru.get_encoder_decoder_from_gru()
        gru_op = gru.predict(noisy_seq,enc,dec)
        model = lstm.get_model_lstm()
        lstm_op = lstm.predict(noisy_seq,model)
        op =torch.mean(torch.stack((gru_op,lstm_op)),dim=0).T
        return self.fc(op)


def train_model(data_inputs,val_inputs,ensemble,optimiser,criterion):
    """
    :param data_inputs: list of tuples of (noise,clean) data used as training
    :param val_inputs: list of tuples of (noise,clean) data used as validation
    :param ensemble: ensemble model of class Ensemble
    :param optimiser: optimiser for training (ex: Adam,SGD)
    :param criterion: criterion for loss backward propagation
    """
    for i in range(epochs):
        ensemble.train()
        for op,ip in data_inputs:
            optimiser.zero_grad()
            op = torch.tensor(op).view(batch_size,seq_len,-1)
            ip = torch.tensor(ip).view(batch_size,seq_len,-1)
            x = ensemble(ip)
            x = x.view(batch_size,seq_len,-1)
            op = op.view(batch_size,seq_len,-1)
            loss = torch.sqrt(criterion(x, op)+eps)
            loss.backward()
            optimiser.step()
        val_loss,_ = evaluate(val_inputs,ensemble,criterion)
        print(f"val loss after epoch {i} : {val_loss}")
        if i%50 == 0 and i >0:
            torch.save({
                'model_state_dict': ensemble.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
            }, path_ensemble)


def evaluate(val_ips,ensemble,criterion):
    """
    :param val_ips: list of tuples containing (noise,clean) data
    :param ensemble: ensemble model of trained gru and lstm encoder decoder models
    :param criterion: loss crtierion (eg. MSELoss, L1Loss)
    :return: loss value, outputs predicted
    """
    outputs = []
    true_ops = []
    ensemble.eval()
    with torch.no_grad():
        for op,ip in val_ips:
            op = torch.tensor(op).view(batch_size,seq_len,-1)
            ip = torch.tensor(ip).view(batch_size,seq_len,-1)
            x = ensemble(ip).view(batch_size,seq_len,-1)
            outputs.append(x)
            true_ops.append(op)
        outputs = torch.cat(outputs,dim=0)
        true_ops = torch.cat(true_ops,dim=0)
        loss = torch.sqrt(criterion(outputs, true_ops)+eps)
        return loss.item(),outputs

