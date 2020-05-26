import torch
import torch.nn as nn 
import numpy as np
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(hidden_size*2, output_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(output_size*2, output_size)
        
        # initializing weights
        nn.init.xavier_normal(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_normal(self.lstm.weight_hh_l0, gain=np.sqrt(2))
       
    def forward(self, encoded_input):
        output, hidden = self.lstm(encoded_input)
        decoded_output = self.fc1(output)
        return decoded_output