import torch
import torch.nn as nn
import numpy as np


# Positional Encoding function
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)


class StandardTransformerModel(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, input_vocab_size, target_vocab_size, seq_length, 
                 device='cuda'):
        super(StandardTransformerModel, self).__init__()
        self.seq_length = seq_length
        self.device = device

        self.src_embedding = nn.Embedding(input_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(target_vocab_size, d_model)
        
        self.pos_encoding = positional_encoding(seq_length, d_model).to(device)
        
        # Create masks once during initialization
        self.causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, target_vocab_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src, tgt, use_teacher_forcing=True):
        if use_teacher_forcing:
            tgt_input = tgt[:, :-1]
        else:
            tgt_input = tgt

        # Create causal mask for decoder
        tgt_mask = self.causal_mask[:tgt_input.size(1), :tgt_input.size(1)].to(self.device)

        # Embedding and positional encoding
        src_emb = self.src_embedding(src) + self.pos_encoding[:, :src.size(1), :]
        tgt_emb = self.tgt_embedding(tgt_input) + self.pos_encoding[:, :tgt_input.size(1), :]
        
        # Transformer layers
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        return self.fc(output)

    # Auto-regressively generate the predictions without teacher forcing.
    def generate(self, X_batch, SOS_token):
        batch_size = X_batch.size(0)
        # Start with SOS token
        decoded = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=self.device)
        
        # Generate exactly seq_length tokens (excluding SOS)
        for i in range(self.seq_length):
            output = self.forward(X_batch, decoded, use_teacher_forcing=False)
            next_token = output[:, -1].argmax(dim=1)

            decoded = torch.cat([decoded, next_token.unsqueeze(1)], dim=1)
            
        predictions = decoded[:, :self.seq_length]

        return predictions
    
    def loss(self, preds, targets):
        tgt = targets[:, 1:]

        # Reshape output and target for loss computation
        preds_flat = preds.contiguous().view(-1, preds.shape[-1])
        target_flat = tgt.contiguous().view(-1)
        
        return self.criterion(preds_flat, target_flat)