import torch,os
from torch import nn
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
from model_building import n_features,learning_rate,hidden_dim,batch_size,seq_len,device,PATH
from model_building.lstm_encoder_decoder import Encoder,Decoder, Seq2Seq
from model_building.train_model import evaluate
from data_preprocessing.filter_ameri_files import get_sequence, add_noise
from model_building.test_data import *
os.chdir("/home/sonakshireddy/Documents/lstm_vis/")


def predictions_plot(result_flat, test, model_name, test_name):
    plt.figure(figsize=(25, 10))
    fig = plt.gcf()
    plt.xlabel('date')
    plt.ylabel('soil moisture')
    plt.plot(test.index[:result_flat.size()[0]]
             , test[:result_flat.size()[0]], color='orange', label='test input')
    plt.plot(test.index[:result_flat.size()[0]]
             , result_flat, label='prediction')
    plt.legend(loc='upper right', shadow=True, fontsize='xx-large')
    fig.savefig(model_name + "_" + test_name + ".png")


def get_model_diff(result_flat, test, model_name, test_name):
    res = pd.Series(result_flat.numpy().reshape(-1), index=test.index[:result_flat.size()[0]])
    actual = test[:result_flat.size()[0]]
    height = (res[actual.index[actual > 0]] - actual[actual.index[actual > 0]])
    plt.figure(figsize=(25, 10))
    fig = plt.gcf()
    plt.xticks(fontsize=14, rotation=45)
    plt.xlabel('date')
    plt.ylabel('prediction - actual')
    plt.bar(actual.index[actual > 0], height)
    fig.savefig(model_name + "_" + test_name + "_model_diff.png")


if __name__ == '__main__':

    # PATH ="/home/sonakshireddy/Documents/soil_ae_models/ae_24_seqlen_batch_1_epoch_100_hidden_256_lr_0.0001_te_0.5_no_att_with_condition_lay=1_noise_0.5_bar_slt_ton=2016.pkl"
    # PATH ="/home/sonakshireddy/Documents/soil_ae_models/ae_24_seqlen_batch_1_epoch_100_hidden_256_lr_0.0001_te_0.5_no_att_with_condition_lay=1_noise_0.5_all_ameri_GRU.pkl"
    model_name = PATH.split("/")[-1].split(".pkl")[0]
    checkpoint = torch.load(PATH)
    embedding_dim = 256
    encoder = Encoder(n_features,seq_len=seq_len,hidden_dim= hidden_dim,batch_size= batch_size).to(device)
    decoder = Decoder(seq_len=seq_len, hidden_dim=hidden_dim, n_features=n_features, batch_size=batch_size).to(device)
    model = Seq2Seq(encoder,decoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimiser = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    encoder_optimiser.load_state_dict(checkpoint['e_optimizer_state_dict'])
    decoder_optimiser.load_state_dict(checkpoint['d_optimizer_state_dict'])
    criterion = nn.L1Loss()

    (test,test_seq),test_name = kendall()
    val, outputs = evaluate(model,criterion,test_seq,batch_size,output=True)
    print(val)
    result_flat = torch.tensor(outputs).view(-1, 1)
    predictions_plot(result_flat, test, model_name, test_name)
    get_model_diff(result_flat, test, model_name, test_name)
    print("end")


