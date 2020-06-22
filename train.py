import pandas as pd
from torch.utils.data import DataLoader,Dataset, Subset
import numpy as np
import tft_model
from data_formatters import ts_dataset  
import data_formatters.base
import expt_settings.configs
import importlib
from data_formatters import utils
import torch.optim as optim
import torch
from tqdm import tqdm
import pickle

importlib.reload(tft_model)
importlib.reload(utils)

ExperimentConfig = expt_settings.configs.ExperimentConfig

config = ExperimentConfig('m5', 'outputs')

with open('data_formatter.pkl', 'rb') as input:
    data_formatter = pickle.load(input)

# data_formatter = config.make_data_formatter()
#
#
# print("*** Training from defined parameters for {} ***".format('m5'))
# data_csv_path = '/home/arda/Desktop/thesis/m5_tft_data.csv'
# print("Loading & splitting data...")
# raw_data = pd.read_csv(data_csv_path, index_col=0)
# print("Data loaded...")
# train, valid, test = data_formatter.split_data(raw_data)
# train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()
#
# with open('train.pkl', 'wb') as output:  # Overwrites any existing file.
#     pickle.dump(train, output, pickle.HIGHEST_PROTOCOL)
#
# with open('valid.pkl', 'wb') as output:  # Overwrites any existing file.
#     pickle.dump(valid, output, pickle.HIGHEST_PROTOCOL)
#
# with open('test.pkl', 'wb') as output:  # Overwrites any existing file.
#     pickle.dump(test, output, pickle.HIGHEST_PROTOCOL)

# Sets up default params
fixed_params = data_formatter.get_experiment_params()
params = data_formatter.get_default_model_params()

# with open('data_formatter.pkl', 'wb') as output:  # Overwrites any existing file.
#     pickle.dump(data_formatter, output, pickle.HIGHEST_PROTOCOL)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fixed_params.update(params)
fixed_params['batch_first'] = True
fixed_params['name'] = 'test'
fixed_params['device'] = device
fixed_params['minibatch_size'] = 128

max_samples = 512 * 10
# ds = ts_dataset.TSDataset(fixed_params, max_samples, train)
#
# with open('ts_dataset.pkl', 'wb') as output:  # Overwrites any existing file.
#     pickle.dump(ds, output, pickle.HIGHEST_PROTOCOL)

with open('ts_dataset.pkl', 'rb') as input:
    ds = pickle.load(input)

batch_size=128
loader = DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=True)

with open('m5_dataloader.pkl', 'wb') as output:
    pickle.dump(loader, output, pickle.HIGHEST_PROTOCOL)

model = tft_model.TFT(fixed_params).to(device)

q_loss_func = tft_model.QuantileLoss([0.1,0.5,0.9])
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
epochs=100
losses = []
for i in range(epochs):
    epoch_loss = [] 
    progress_bar = tqdm(enumerate(loader), total=len(loader))
    for batch_num, batch in progress_bar:
        optimizer.zero_grad()
        output, all_inputs, attention_components = model(batch['inputs'])
        loss= q_loss_func(output[:,:,:].view(-1,3), batch['outputs'][:,:,0].flatten().float().to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), fixed_params['max_gradient_norm'])
        optimizer.step()
        epoch_loss.append(loss.item())
    
    losses.append(np.mean(epoch_loss))
    if loss.item() <= min(losses):
        torch.save(model.state_dict(), 'm5_best_model.pth')
        
    print(np.mean(epoch_loss))