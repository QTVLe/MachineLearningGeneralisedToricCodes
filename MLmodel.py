# version 240225 - transformer 2.6, 3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss

# from GPT2mindist_26_GA import GPT2mindist as GPT2mindist_26
# from GPT2mindist_2_GA import GPT2mindist as GPT2mindist_2
from GPT2mindist_3 import GPT2mindist as GPT2mindist_3 

import numpy as np

torch.set_default_dtype(torch.float32)
global_dtype = torch.float32
# global_np_dtype = np.float32

# Custom Dataset
class SequenceDataset(Dataset):
  def __init__(self, codes, mindists):
    self.codes = codes
    self.mindists = mindists

  def __len__(self):
    return len(self.codes)

  def __getitem__(self, index):
    return self.codes[index], self.mindists[index]

# Collate function for padding inputs and stacking outputs
def collate_fn(batch):
    # Separate inputs and outputs
    inputs, outputs = zip(*batch)

    # Pad sequences to the same length
    inputs_padded = pad_sequence(inputs, batch_first=True)  # Shape: (batch_size, max_length, width)

    # Convert outputs to a tensor (assuming outputs are scalar or 1D values)
    outputs_tensor = torch.tensor(outputs)

    # Lengths of each sequence in the batch
    lengths = torch.tensor([len(seq) for seq in inputs])

    return inputs_padded, outputs_tensor, lengths

class LSTM2dist(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM2dist, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        # The LSTM takes lines of code as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0.2, batch_first=True)

        # The linear layer that maps from hidden state space to distance
        self.hidden2out = nn.Linear(hidden_dim + hidden_dim, output_dim) # to take concat of cell and hidden states as an input

    def forward(self, input):#, lengths):
        # input shape (mb, seq_length, input_dim)
        # length shape (mb, seq_length)

        # packed_input = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)

        _, (hidden, cell) = self.lstm(input)

        # Extract hidden state and cell state from the last layer
        final_hidden = hidden[-1]
        final_cell = cell[-1]

        # Concat hidden and cell states
        combined = torch.cat((final_hidden, final_cell), dim=-1)
        output = self.hidden2out(combined)
        return output

# Define RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.output_size = output_size

        self.i2h = nn.Sequential(
            nn.Linear(input_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, hidden_size)
        )
        self.h2h = nn.Sequential(
            nn.Linear(hidden_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, hidden_size)
        )
        self.h2o = nn.Sequential(
            nn.Linear(hidden_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, output_size)
        )

    def forward(self, input):
        # input: shape (max_length, input_size)
        # hidden: shape (hidden_size,)

        hidden = self.initHidden(1)

        for line in input:
            # Update hidden state
            hidden = F.tanh(self.i2h(line) + self.h2h(hidden))

        output = self.h2o(hidden)
        return output

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class MLModel():
    def __init__(self, PATH, model_type = "Transformer"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type

        if model_type == "LSTM":
            # LSTM ML Model parameters
            num_layers = 2
            hidden_dim = 128
            output_dim = 1
            input_dim = 36
            parameters = [input_dim, hidden_dim, output_dim, num_layers]

            checkpoint = torch.load(PATH, weights_only=False)
            self.model = LSTM2dist(*parameters)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.last_test_loss = checkpoint['all_test_losses'][-1]
            self.last_train_loss = checkpoint['all_losses'][-1]
        elif model_type == "RNN":
            # RNN parameters
            input_size = 36
            hidden_size = 128
            layer_size = 256
            output_size = 1
            parameters = [input_size, hidden_size, layer_size, output_size] 

            checkpoint = torch.load(PATH, weights_only=False)
            self.model = RNN(*parameters)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.last_test_loss = checkpoint['all_test_losses'][-1]
            self.last_train_loss = checkpoint['all_losses'][-1]
        # elif model_type == "Transformer_v26":
        #     # Transformer parameters are defined in transformer file
        #     self.model = GPT2mindist_26()
        #     self.model.to(self.device)
        #     if self.device=='cpu':
        #         checkpoint = torch.load(PATH, weights_only=False, map_location=torch.device('cpu'))
        #     else:
        #         checkpoint = torch.load(PATH, weights_only=False)     
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
            
        #     self.last_test_loss = None
        #     self.last_train_loss = None
        # elif model_type == "Transformer_v2":
        #     # Transformer parameters are defined in transformer file
        #     self.model = GPT2mindist_2()
        #     self.model.to(self.device)
        #     if self.device=='cpu':
        #         checkpoint = torch.load(PATH, weights_only=False, map_location=torch.device('cpu'))
        #     else:
        #         checkpoint = torch.load(PATH, weights_only=False)     
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
            
        #     self.last_test_loss = None
        #     self.last_train_loss = None
        elif model_type == "Transformer_v3":
            # Transformer parameters are defined in transformer file
            self.model = GPT2mindist_3()
            self.model.to(self.device)
            if self.device=='cpu':
                checkpoint = torch.load(PATH, weights_only=False, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(PATH, weights_only=False)     
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.last_test_loss = None
            self.last_train_loss = None
        else:
            raise ValueError("Input correct model type! Shpuld be \"LSTM\" or \"RNN\".")
    
        self.model.eval()

        self.criterion = mse_loss

        # Denormalise ML model outputs (LSTM/RNN)
        self.mean = torch.tensor((30.+3.)/2.)
        self.std = torch.tensor((30.-3.)/2.)

    def denormalize(self, mindist):
        return mindist * self.std + self.mean
    
    def predict(self, matrix):
        '''takes toric code matrix as numpy array and outputs predition as float'''
        if isinstance(matrix, np.ndarray):
            torch_matrix = torch.from_numpy(matrix)
        elif torch.is_tensor(matrix):
            torch_matrix = matrix
        else:
            raise TypeError("Input is neither numpy.ndarray, nor torch.Tensor")
        
        if self.model_type == "Transformer_v26" or self.model_type == "Transformer_v2":
            if len(torch_matrix.shape)<3:
                code = torch.unsqueeze(torch_matrix, dim=0) # add batch dimension
            length = torch.tensor([code.shape[1]], dtype=torch.long)

            classes_all = torch.arange(36, dtype=global_dtype)
            logits, _ = self.model(code, length)
            prob = F.softmax(logits, dim=1).cpu() # shape (B, n_output)
            return torch.squeeze(prob @ classes_all).detach().numpy()
        elif self.model_type == "Transformer_v3":
            if len(torch_matrix.shape)<3:
                code = torch.unsqueeze(torch_matrix, dim=0) # add batch dimension
            length = torch.tensor([code.shape[1]], dtype=torch.long)

            classes_all = torch.arange(36, dtype=global_dtype)
            prob, _ = self.model(code, length)
            return torch.squeeze(prob @ classes_all).detach().numpy()
        else:
            prediction = self.model(torch_matrix)
            return float(self.denormalize(prediction.item()))
    
    def evaluate(self, dataset, normalised = True):
        losses = torch.zeros(len(dataset))
        with torch.no_grad():
            for i, data in enumerate(dataset):
                code, mindist = data
                out_mindist = torch.flatten(self.model(code))

                if normalised:
                    losses[i] = self.criterion(self.denormalize(out_mindist), self.denormalize(mindist)).item()
                else:
                    losses[i] = self.criterion(self.denormalize(out_mindist), mindist).item()

        return losses.mean()
    
def load_dataset(codes_dir, mindist_dir):
    with open(codes_dir,'rb') as file:
        codes_data = torch.load(file)

    with open(mindist_dir,'rb') as file:
        mindist_data = torch.load(file)

    # loading from file changes dtype for some reason
    codes_data = list(map(lambda x: x.to(global_dtype), codes_data))
    mindist_data = list(map(lambda x: x.to(global_dtype), mindist_data))

    dataset_len = len(codes_data)

    # Normalize output
    mean = torch.tensor((30.+3.)/2.)
    std = torch.tensor((30.-3.)/2.)
    for i in range(len(mindist_data)):
        mindist_data[i] = (mindist_data[i] - mean)/std

    split_len = int(dataset_len * 0.9)
    trainset = SequenceDataset(codes_data[:split_len], mindist_data[:split_len])
    testset = SequenceDataset(codes_data[split_len:], mindist_data[split_len:])

    return trainset, testset