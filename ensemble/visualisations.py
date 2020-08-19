import pandas as pd
import numpy as np
import torch,os
from torch import nn
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import time,random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('seaborn')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
LARGE_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)
from model_building import n_features,learning_rate,hidden_dim,batch_size,seq_len,device
from model_building.test_data import *
from model_building.visualisations import predictions_plot,get_model_diff
from ensemble.lstm_predict import get_model_lstm
from ensemble.gru_predict import get_encoder_decoder_from_gru
from ensemble.ensemble_model import Ensemble,evaluate
from ensemble import path_ensemble
os.chdir("/home/sonakshireddy/Documents/ensemble/")

PATH = path_ensemble


model_name = PATH.split("/")[-1].split(".pkl")[0]
checkpoint = torch.load(PATH)
embedding_dim = 256

gru_enc,gru_dec = get_encoder_decoder_from_gru()
lstm_model = get_model_lstm()
ensemble_model = Ensemble(lstm_model,gru_enc,gru_dec)
ensemble_model.load_state_dict(checkpoint['model_state_dict'])
opt = torch.optim.Adam(ensemble_model.parameters(), lr=learning_rate)
opt.load_state_dict(checkpoint['optimizer_state_dict'])
criterion = nn.MSELoss()

(test, test_seq), test_name = scd()
val, outputs = evaluate(test_seq,ensemble_model, criterion)
print(np.sqrt(val))
result_flat = outputs.view(-1, 1)
predictions_plot(result_flat, test, model_name, test_name)
get_model_diff(result_flat, test, model_name, test_name)
print("end")