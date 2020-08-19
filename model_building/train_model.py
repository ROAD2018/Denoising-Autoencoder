import torch, random, copy
from torch import nn
import numpy as np
from model_building import *


def train(model, optimiser, criterion, train_dataset, batch_size):
    """
    :param model: Seq2Seq model to train the denoising of the soil moisture data
    :param optimiser: optimiser for training (ex: Adam,SGD)
    :param criterion: criterion for loss backward propagation
    :param train_dataset: list of tuples of (noise,clean) data used as training
    :param batch_size: int - batch size of the input data to the model
    :return: mean of train losses for all the data points in train_dataset
    """
    model = model.train()
    train_losses = []
    for seq_true, noisy_true in train_dataset:
        optimiser.zero_grad()
        seq_true = (seq_true).view(-1, batch_size).to(device)
        noisy_data = (noisy_true).view(-1, batch_size).to(device)
        #             print(seq_true.size())
        decoder_ops = model(noisy_data, seq_true)
        loss = criterion(decoder_ops.view(-1, batch_size), seq_true.view(-1, batch_size))
        loss.backward()
        optimiser.step()
        train_losses.append(loss.item())
    return np.mean(train_losses)


def evaluate(model, criterion, val_dataset, batch_size, output=False):
    """
    :param model: model to train on
    :param criterion: criterion for loss (eg. MSE, L1)
    :param val_dataset: validation dataset
    :param batch_size: int - batch size of each input
    :param output: boolean flag to return outputs after evaluation
    :return: validation loss, outputs predicted
    """
    val_losses = []
    outputs = []
    model.eval()
    with torch.no_grad():
        for seq_true, noisy_data in val_dataset:
            seq_true = (seq_true).view(-1, batch_size)
            noisy_data = torch.tensor(noisy_data).view(-1, batch_size)
            decoder_ops = model(noisy_data, seq_true)
            loss = criterion(decoder_ops.view(-1, batch_size), seq_true.view(-1, batch_size))
            val_losses.append(loss.item())
            if output:
                outputs.append(decoder_ops.view(batch_size, -1, 1).detach().numpy())
    return np.mean(val_losses), outputs


def train_model(model, optimiser, criterion, train_dataset, val_dataset,
                batch_size=32, n_epochs=100):
    """
    :param model: model to train on
    :param optimiser: optimiser for model (eg. Adam, SGD)
    :param criterion: criterion for loss (eg. MSE, L1)
    :param train_dataset: training dataset
    :param val_dataset: validation dataset
    :param batch_size: int - batch size of each input
    :param n_epochs: int - number of epochs to train the model
    :return: best loss after training the model, dict containing history of the losses
    """
    global teacher_forcing_ratio
    history = dict(train=[], val=[])
    init_tf_ratio = teacher_forcing_ratio
    best_loss = 10000.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimiser = copy.deepcopy(optimiser.state_dict())
    for epoch in range(1, n_epochs + 1):
        train_loss = train(model, optimiser, criterion, train_dataset, batch_size)
        val_loss, _ = evaluate(model, criterion, val_dataset, batch_size)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_optimiser = copy.deepcopy(optimiser.state_dict())
        print(
            f'Epoch {epoch}: train loss {train_loss} val loss {val_loss} \n')
        if epoch != 0 and epoch % 50 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
            }, PATH)
        teacher_forcing_ratio = init_tf_ratio * (1 / (1 + decay * epoch))
    model.load_state_dict(best_model_wts)
    optimiser.load_state_dict(best_optimiser)
    return best_loss, history
