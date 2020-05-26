import torch
import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self, input_size, num_layers, max_words):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_size, 1, num_layers, batch_first=True)
        self.fc1 = nn.Linear(max_words, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        output, hidden = self.lstm(input)
        output = torch.flatten(output)
        output = self.fc1(output)
        output = self.sigmoid(output)
        return output



"""
class Discriminator(nn.Module):
    def __init__(self, input_size, num_layers, max_words, batch_size):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_size, 1, num_layers, batch_first=True)
        self.fc1 = nn.Linear(batch_size*max_words, batch_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        output, hidden = self.lstm(input)
        output = torch.flatten(output)
        output = self.fc1(output)
        output = self.sigmoid(output)
        return output

"""