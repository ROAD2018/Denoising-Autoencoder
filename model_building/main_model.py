import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from model_building import *
from model_building.lstm_encoder_decoder import Encoder,Decoder, Seq2Seq
from model_building.train_model import train_model

input_sequence = np.load("/home/sonakshireddy/Downloads/zips/input_bst.npy")
noisy_sequence = np.load("/home/sonakshireddy/Downloads/zips/noise_bst.npy")
# val_input = np.load("/home/sonakshireddy/Downloads/zips/val_input_all_ameri.npy")
# val_noisy = np.load("/home/sonakshireddy/Downloads/zips/val_noise_all_ameri.npy")

# train_dataset = DataLoader(list(zip(input_sequence,noisy_sequence)), shuffle=False)
# val_dataset = DataLoader(list(zip(val_input,val_noisy)), shuffle=False)
train_dataset, val_dataset = train_test_split(
  list(zip(input_sequence,noisy_sequence)),
  test_size=0.12465753424,
  random_state=RANDOM_SEED
)
criterion = nn.L1Loss()
encoder = Encoder(n_features,seq_len=seq_len,hidden_dim= hidden_dim,batch_size= batch_size).to(device)
decoder = Decoder(seq_len=seq_len, hidden_dim=hidden_dim, n_features=n_features, batch_size=batch_size).to(device)
model = Seq2Seq(encoder,decoder).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
h = train_model(model=model,optimiser=optimiser,criterion=criterion,train_dataset=train_dataset,val_dataset=val_dataset,batch_size=batch_size,n_epochs=epochs)
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            }, PATH)