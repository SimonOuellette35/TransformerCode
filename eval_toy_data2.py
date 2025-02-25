import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.transformer_model import StandardTransformerModel

# ======================================================= Setup & Hyper-parameters =========================================================

# Set deterministic seed for reproducibility
DET_SEED = 123
torch.manual_seed(DET_SEED)
torch.cuda.manual_seed(DET_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(DET_SEED)


# Hyperparameters
num_epochs = 300
batch_size = 50
d_model = 64   # Transformer model size
num_heads = 4
num_layers = 2
num_classes = 21  # Output vocabulary size (+1 for SOS token)
learning_rate = 0.0001
SOS_token = num_classes - 1  # Define start-of-sequence token
max_seq_length = 30

trainN = 5000
valN = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================================= Toy problem data generation =====================================================

# Generate toy sequential data
# This dataset works as follows:
# X is random tokens of varying sequence lengths with 0 padding.
# Y is X where each token is repeated twice in a row.
def generate_data(num_samples=1000):
    # Generate random sequence lengths between 1 and seq_length
    lengths = torch.randint(1, max_seq_length//2, (num_samples,))
    
    # Initialize tensors with padding (0)
    X = torch.zeros((num_samples, max_seq_length), dtype=torch.long)
    Y = torch.zeros((num_samples, max_seq_length), dtype=torch.long)
    
    # Fill tensors according to random lengths
    for i in range(num_samples):
        # Generate random tokens for this sequence
        seq = torch.randint(1, num_classes-1, (lengths[i],))  # Start from 1 to reserve 0 for padding
        X[i, :lengths[i]] = seq
        
        # Create Y by repeating each token twice
        repeated = seq.repeat_interleave(2)  # Repeat each element twice
        
        # Add SOS token at start and truncate if too long
        Y[i, 0] = SOS_token
        Y[i, 1:min(len(repeated)+1, max_seq_length)] = repeated[:max_seq_length-1]  # Leave room for SOS
        
    return X, Y

X_train, Y_train = generate_data(trainN)
X_val, Y_val = generate_data(valN)
dataset_train = TensorDataset(X_train, Y_train)
dataset_val = TensorDataset(X_val, Y_val)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size)

# ========================================================= Model initialization & training =================================================

def evaluate(dataloader):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                predictions = model.generate(X_batch, SOS_token)
                correct += (predictions == Y_batch).sum().item()
                total += X_batch.numel()

    return (correct / total) * 100

# Initialize model
model = StandardTransformerModel(d_model, 
                                 num_heads, 
                                 num_layers, 
                                 num_classes, 
                                 num_classes,
                                 max_seq_length).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training for {num_epochs} epochs on a total of {trainN} samples...")
# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch, Y_batch)

        loss = model.loss(output, Y_batch)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        accuracy = evaluate(val_loader)
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}, Validation Accuracy: {accuracy:.2f}%")
