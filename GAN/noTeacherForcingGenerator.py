import torch
import torch.nn as nn 

class Generator(nn.Module):
    def __init__(self, encoder, decoder):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, input):
        encoded_input, original_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output