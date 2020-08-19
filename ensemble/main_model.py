from torch import nn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from ensemble.gru_predict import get_encoder_decoder_from_gru
from ensemble.lstm_predict import get_model_lstm
from ensemble.ensemble_model import Ensemble,train_model
from model_building import learning_rate, RANDOM_SEED
from ensemble import path_ensemble


input_sequence = np.load("/home/sonakshireddy/Downloads/zips/input_all_ameri1.npy")
noisy_sequence = np.load("/home/sonakshireddy/Downloads/zips/noise_all_ameri1.npy")
# input_sequence = input_sequence[:50]
# noisy_sequence = noisy_sequence[:50]
# val_input = val_input[:10]
# val_noisy = val_noisy[:10]
train_dataset, val_dataset = train_test_split(
  list(zip(input_sequence,noisy_sequence)),
  test_size=0.12465753424,
  random_state=RANDOM_SEED
)
gru_enc,gru_dec = get_encoder_decoder_from_gru()
lstm_model = get_model_lstm()
ensemble_model = Ensemble(lstm_model,gru_enc,gru_dec)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(ensemble_model.parameters(), lr=learning_rate)
train_model(data_inputs=train_dataset,val_inputs=val_dataset,ensemble=ensemble_model,optimiser=optimiser,criterion=criterion)
torch.save({
                'model_state_dict': ensemble_model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
            }, path_ensemble)


