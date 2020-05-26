import torch
import torch.nn as nn 
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.lrelu = nn.LeakyReLU()
        
        # initializing weights
        nn.init.xavier_normal(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_normal(self.lstm.weight_hh_l0, gain=np.sqrt(2))
         
    def forward(self, input):
        original_input = input
        input = self.dropout(input)
        encoded_input, hidden = self.lstm(input)
        encoded_input = self.fc1(encoded_input)
        encoded_input = self.lrelu(encoded_input)
        return encoded_input, original_input