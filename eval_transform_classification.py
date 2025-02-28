import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.transformer_model import StandardTransformerModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json


# ======================================================= Setup & Hyper-parameters =========================================================

# Set deterministic seed for reproducibility
DET_SEED = 123
torch.manual_seed(DET_SEED)
torch.cuda.manual_seed(DET_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(DET_SEED)


# Hyperparameters
num_epochs = 5000
batch_size = 50
d_model = 256
num_heads = 16
num_layers = 4
input_vocab_size = 13
num_classes = 4
learning_rate = 0.0001  # Slightly increased initial learning rate
SOS_token = num_classes - 1  # Define start-of-sequence token
max_seq_length = 86

trainN = 10000
valN = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================================= Toy problem data generation =====================================================

# We generate 3 types of input grid to output grid transformations:
# 1. vmirror + change non-black pixels to yellow
# 2. hmirror + change non-black pixels to yellow
# 3. shift pixels to the right

# The goal of this datasets is that the model must learn to classify which type of operation was applied between
# the two input grids.

def load_data(num_samples=1000, filename='training.json'):
    X_train = []
    Y_train = []

    try:
        with open(filename, 'r') as f:
            data_list = json.load(f)
            # Take only up to num_samples
            data_list = data_list[:num_samples]
            
            for data in data_list:
                X_train.append(data['input_sequence'])
                Y_train.append(data['task_desc'])
                
    except json.JSONDecodeError:
        # Fallback for line-delimited JSON
        with open(filename, 'r') as f:
            # Read up to num_samples lines
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                    
                data = json.loads(line)
                X_train.append(data['input_sequence'])
                Y_train.append(data['task_desc'])
            
    # Convert to PyTorch tensors
    X_train = torch.tensor(np.array(X_train), dtype=torch.long)
    Y_train = torch.tensor(np.array(Y_train), dtype=torch.long)
    
    return X_train, Y_train

# Load data with a progress bar
print("Loading training data...")
X_train, Y_train = load_data(trainN, 'training.json')
print("Loading validation data...")
X_val, Y_val = load_data(valN, 'validation.json')

# Check class distribution
train_class_counts = {}
for y in Y_train[:, 1].numpy():
    train_class_counts[y] = train_class_counts.get(y, 0) + 1
print(f"Training class distribution: {train_class_counts}")

val_class_counts = {}
for y in Y_val[:, 1].numpy():
    val_class_counts[y] = val_class_counts.get(y, 0) + 1
print(f"Validation class distribution: {val_class_counts}")

dataset_train = TensorDataset(X_train, Y_train)
dataset_val = TensorDataset(X_val, Y_val)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size)

# ========================================================= Model initialization & training =================================================

def evaluate(dataloader, verbose=False):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in dataloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                predictions = model.generate(X_batch, SOS_token)

                # Trim predictions to match target length
                predictions = predictions[:, :Y_batch.size(1)]

                # Compare predictions with targets element-wise
                matches = (predictions == Y_batch)
                
                # Count total correct predictions
                tmp_correct = matches.sum().item()
                correct += tmp_correct
                total += Y_batch.numel()
            
            if verbose:
                # Find first batch element with incorrect prediction
                for batch_idx in range(predictions.size(0)):
                    batch_matches = matches[batch_idx]
                    if not batch_matches.all():
                        print(f"\nFirst incorrect prediction found in batch element {batch_idx}")
                        print(f"Predicted flip: {predictions[batch_idx][1].item()}")
                        print(f"Ground truth flip: {Y_batch[batch_idx][1].item()}")
                        # Get the incorrect example
                        x = X_batch[batch_idx].cpu().numpy()
                        y = Y_batch[batch_idx].cpu().numpy()
                        
                        # Split into original sequences (first and second half)
                        max_len = x.shape[0] // 2
                        input_seq = x[:max_len]
                        target_seq = x[max_len:]
                        
                        # Convert sequences back to grids
                        def seq_to_grid(seq):
                            grid = []
                            current_row = []
                            for token in seq:
                                if token == 1:  # Row marker
                                    if current_row:  # Only append if row is not empty
                                        grid.append(current_row)
                                        current_row = []
                                elif token == 2:  # EOS token
                                    if current_row:  # Only append if row is not empty
                                        grid.append(current_row)
                                    break
                                elif token == 0:  # PAD token
                                    continue
                                elif token >= 3 and token <= 12:  # Pixel colors
                                    current_row.append(token)
                            return np.array(grid)
                        
                        input_grid = seq_to_grid(input_seq)
                        target_grid = seq_to_grid(target_seq)
                        
                        # Create subplots
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                        
                        # Plot input grid
                        ax1.imshow(input_grid, cmap='binary')
                        ax1.set_title('Input Grid')
                        ax1.axis('off')
                        
                        # Plot target grid
                        ax2.imshow(target_grid, cmap='binary')
                        ax2.set_title(f'Target Grid (Flip: {y[1]})')
                        ax2.axis('off')
                        
                        plt.tight_layout()
                        plt.show()
                        break

    return (correct / total) * 100

# Initialize model with dropout
model = StandardTransformerModel(d_model, 
                                 num_heads, 
                                 num_layers, 
                                 input_vocab_size, 
                                 num_classes,
                                 max_seq_length,
                                 dropout=0.2).to(device)  # Added dropout parameter

# Calculate and display the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Model has {total_params:,} trainable parameters")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Add learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, 
                             verbose=True, min_lr=1e-6)

print(f"Training for {num_epochs} epochs on a total of {trainN} samples...")
train_losses = []
# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch, Y_batch)

        loss = model.loss(output, Y_batch)

        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss/len(train_loader)
    train_losses.append(avg_loss)
    
    # Update learning rate based on loss
    scheduler.step(avg_loss)
    
    if epoch % 100 == 0:
        val_accuracy = evaluate(val_loader)
        train_accuracy = evaluate(train_loader)

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}, Train. Accuracy: {train_accuracy:.2f}%, Val. Accuracy: {val_accuracy:.2f}%")
    else:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")
