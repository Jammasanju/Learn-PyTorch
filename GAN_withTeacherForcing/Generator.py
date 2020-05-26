import torch
import torch.nn as nn 
import random

class Generator(nn.Module):
    def __init__(self, encoder, decoder, max_words, hidden_size, embedding_length):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_words = max_words
        self.hidden_size = hidden_size
        self.emb_length = embedding_length
        
    def forward(self, input, teacher_forcing_ratio):
        encoded_input, origianl_input = self.encoder(input)
        
        batch_size = encoded_input.size(0)
        outputs = torch.zeros(batch_size, self.max_words, self.emb_length)
        
        for batch_index, batch in enumerate(encoded_input):
            decoded = torch.zeros(self.max_words, self.emb_length)
            
            for hidden_index, hidden_embedding in enumerate(batch):
                decoder_out = self.decoder(hidden_embedding.view(1,1,self.hidden_size))
                teacher_force = random.random() < teacher_forcing_ratio
                if(teacher_force):
                    decoded[hidden_index] = origianl_input[batch_index][hidden_index]
                else:
                    decoded[hidden_index] = decoder_out.view(self.emb_length)
                
            outputs[batch_index] = decoded
        return outputs