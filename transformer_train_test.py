import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Hyperparameters
num_epochs = 250
batch_size = 16
seq_length = 10
d_model = 32   # Transformer model size
num_heads = 4
num_layers = 2
num_classes = 21  # Output vocabulary size (+1 for SOS token)
learning_rate = 0.001
SOS_token = num_classes - 1  # Define start-of-sequence token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate toy sequential data
def generate_data(num_samples=1000):
    X = torch.randint(0, num_classes - 1, (num_samples, seq_length))  # Exclude SOS from inputs
    Y = torch.cat([torch.full((num_samples, 1), SOS_token), X[:, :-1]], dim=1)  # Shift right with SOS
    return X, Y

X_train, Y_train = generate_data(5000)
X_val, Y_val = generate_data(1000)

dataset_train = TensorDataset(X_train, Y_train)
dataset_val = TensorDataset(X_val, Y_val)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size)

# Positional Encoding Class
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

class TransformerModel(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, num_classes, seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_classes, d_model)
        self.pos_encoding = positional_encoding(seq_length, d_model).to(device)
        
        # Create masks once during initialization
        self.causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, tgt):
        # Create src padding mask (not needed for this toy example but good practice)
        src_key_padding_mask = None
        
        # Create causal mask for decoder
        tgt_mask = self.causal_mask[:tgt.size(1), :tgt.size(1)].to(device)
        tgt_key_padding_mask = None
        
        # Embedding and positional encoding
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]
        
        # Transformer layers
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb, memory, 
                            tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask)
        
        return self.fc(output)

# Initialize model
model = TransformerModel(d_model, num_heads, num_layers, num_classes, seq_length).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch, Y_batch)
        # Reshape output and target for loss computation
        output_flat = output.view(-1, num_classes)
        target_flat = X_batch.view(-1)
        loss = criterion(output_flat, target_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Debug function to print tensor shapes
def print_shape(name, tensor):
    print(f"{name} shape: {tensor.shape}")

# Autoregressive Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(device)
        batch_size = X_batch.size(0)
        
        # Start with SOS token
        decoded = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)
        
        # Generate exactly seq_length tokens (excluding SOS)
        for i in range(seq_length):
            print_shape("Current decoded", decoded)
            output = model(X_batch, decoded)
            print_shape("Model output", output)
            next_token = output[:, -1].argmax(dim=1)
            print_shape("Next token", next_token)
            decoded = torch.cat([decoded, next_token.unsqueeze(1)], dim=1)
            print_shape("Updated decoded", decoded)
            print(f"Generation step {i+1}/{seq_length}")
            
        # Now decoded should have seq_length + 1 tokens (SOS + seq_length predictions)
        print_shape("Final decoded", decoded)
        print_shape("X_batch", X_batch)
        
        # Take exactly seq_length tokens after SOS
        predictions = decoded[:, 1:seq_length+1]
        print_shape("Predictions", predictions)
        print_shape("Targets", X_batch)
        
        correct += (predictions == X_batch).sum().item()
        total += X_batch.numel()

print(f"Validation Accuracy: {correct / total:.4f}")