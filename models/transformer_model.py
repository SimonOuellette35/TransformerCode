import torch
import torch.nn as nn
import numpy as np
import math


# Positional Encoding function
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

# RoPE positional encoding (Rotary Position Encoding)
# TODO: I'm not sure this is implemented correctly because it doesn't seem to perform as 
# well as the standard positional encoding above. Slower training convergence.
def get_RoPE_embeddings(seq_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    sinusoid_inp = torch.einsum("ij,k->ik", pos, inv_freq)
    embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
    return embeddings

def apply_RoPE(x, positional_embeddings):
    # Slice positional embeddings to match input sequence length
    seq_len = x.size(1)
    positional_embeddings = positional_embeddings[:seq_len, :]
    # Expand positional embeddings to match batch dimension
    positional_embeddings = positional_embeddings.expand(x.size(0), -1, -1)
    # Element-wise multiplication
    x_rotated = x * positional_embeddings
    return x_rotated

class StandardTransformerModel(nn.Module):

    def __init__(self, d_model, num_heads, num_layers, input_vocab_size, target_vocab_size, seq_length,
                 use_rope_embedding=False, device='cuda', dropout=0.1):
        super(StandardTransformerModel, self).__init__()
        self.seq_length = seq_length
        self.device = device
        self.use_rope_embedding = use_rope_embedding
        self.d_model = d_model

        # Initialize embeddings with better scaling
        self.src_embedding = nn.Embedding(input_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(target_vocab_size, d_model)
        
        # Initialize embeddings with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.src_embedding.weight, gain=1.0)
        nn.init.xavier_uniform_(self.tgt_embedding.weight, gain=1.0)
        
        # Scale embeddings by sqrt(d_model)
        self.embed_scale = math.sqrt(d_model)
        
        if self.use_rope_embedding:
            self.pos_encoding = get_RoPE_embeddings(seq_length, d_model).to(device)
        else:
            self.pos_encoding = positional_encoding(seq_length, d_model).to(device)

        # Create masks once during initialization
        self.causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Create encoder and decoder with improved parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 4,  # Increased feedforward dimension
            dropout=dropout,
            activation='gelu',  # Using GELU activation instead of ReLU
            batch_first=True,
            norm_first=True  # Pre-normalization for better training stability
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        # Add layer normalization
        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=decoder_norm)
        
        self.fc = nn.Linear(d_model, target_vocab_size)
        # Initialize output projection with Xavier/Glorot
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing

    def forward(self, src, tgt, use_teacher_forcing=True):
        if use_teacher_forcing:
            tgt_input = tgt[:, :-1]
        else:
            tgt_input = tgt

        # Create causal mask for decoder
        tgt_mask = self.causal_mask[:tgt_input.size(1), :tgt_input.size(1)].to(self.device)

        # Embedding and positional encoding with scaling
        src_emb = self.src_embedding(src) * self.embed_scale
        tgt_emb = self.tgt_embedding(tgt_input) * self.embed_scale
        
        if self.use_rope_embedding:
            src_emb = apply_RoPE(src_emb, self.pos_encoding)
            tgt_emb = apply_RoPE(tgt_emb, self.pos_encoding)
        else:
            src_emb = src_emb + self.pos_encoding[:, :src_emb.size(1), :]
            tgt_emb = tgt_emb + self.pos_encoding[:, :tgt_emb.size(1), :]
        
        # Apply dropout after embedding and positional encoding
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        # Transformer layers
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        return self.fc(output)

    # Auto-regressively generate the predictions without teacher forcing.
    def generate(self, X_batch, SOS_token):
        batch_size = X_batch.size(0)
        # Start with SOS token
        decoded = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=self.device)
        
        # For classification tasks, we only need to generate a few tokens
        max_gen_length = 2  # For SOS + class label
        
        # Generate tokens auto-regressively
        for i in range(max_gen_length - 1):  # -1 because we already have SOS
            # Forward pass through the model
            output = self.forward(X_batch, decoded, use_teacher_forcing=False)
            
            # Get the most likely next token
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=1)
            
            # Append the predicted token to the sequence
            decoded = torch.cat([decoded, next_token.unsqueeze(1)], dim=1)
        
        # For debugging: print confidence scores for the classification
        if batch_size > 0 and max_gen_length > 1:
            # Get the logits for the first prediction after SOS
            class_logits = self.forward(X_batch, decoded[:, :1], use_teacher_forcing=False)[:, 0, :]
            class_probs = torch.nn.functional.softmax(class_logits, dim=1)
            
            # For the first example in the batch
            if False:  # Set to True for debugging
                print("Classification probabilities for first example:")
                for i in range(class_probs.size(1)):
                    print(f"Class {i}: {class_probs[0, i].item():.4f}")
        
        return decoded

    def loss(self, preds, targets):
        tgt = targets[:, 1:]

        # Reshape output and target for loss computation
        preds_flat = preds.contiguous().view(-1, preds.shape[-1])
        target_flat = tgt.contiguous().view(-1)
        
        return self.criterion(preds_flat, target_flat)
        