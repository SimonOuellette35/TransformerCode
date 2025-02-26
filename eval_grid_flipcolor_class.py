import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.transformer_model import StandardTransformerModel
import matplotlib.pyplot as plt

# ======================================================= Setup & Hyper-parameters =========================================================

# Set deterministic seed for reproducibility
DET_SEED = 123
torch.manual_seed(DET_SEED)
torch.cuda.manual_seed(DET_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(DET_SEED)


# Hyperparameters
num_epochs = 1000
batch_size = 50
d_model = 64   # Transformer model size
num_heads = 8
num_layers = 2
input_vocab_size = 13  # 0-9 digits, row marker (10), EOS (11), PAD (12)
num_classes = 3  # Output vocabulary size (+1 for SOS token)
learning_rate = 0.0001
SOS_token = num_classes - 1  # Define start-of-sequence token
max_seq_length = 86  # Updated to match Y tensor size (43 + 1 for SOS token)

trainN = 20000
valN = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================================= Toy problem data generation =====================================================

# Generate toy grid data
# This dataset works as follows:
# X contains 5x5 or 6x6 grids with random integers from 0-9
# Y contains the same grid flipped either vertically or horizontally, randomly
# Each row ends with token 10, and sequence ends with token 11 (EOS)
# For 5x5 grids, remaining tokens are padded with 12 (PAD)
# The task is to classify whether Y is a horizontal or vertical flip of X
def generate_data(num_samples=1000):
    # Randomly choose between 5x5 and 6x6 grids for each sample
    grid_sizes = torch.randint(5, 7, (num_samples,))
    
    # Calculate max sequence length needed (including row markers and EOS)
    # For a NxN grid: N rows * (N cols + 1 row marker) + 1 EOS token
    max_len = 6 * 7 + 1  # Maximum possible length (6x6 grid)
    
    # Initialize tensors with padding token 12
    X = torch.full((num_samples, max_len), 12, dtype=torch.long)
    Y = torch.full((num_samples, max_len), 12, dtype=torch.long)  # Intermediate tensor for rotated grid
    combined_input = torch.full((num_samples, 2*max_len), 12, dtype=torch.long)  # Will hold concatenated X and Y
    labels = torch.full((num_samples, 2), 12, dtype=torch.long)  # +1 for SOS token
    
    # Generate data for each sample
    i = 0
    while i < num_samples:
        N = grid_sizes[i].item()

        # Generate grid with at least 4 non-zero cells
        grid = torch.zeros((N, N), dtype=torch.long)
        num_nonzero = torch.randint(4, N*N+1, (1,)).item()  # Random number between 4 and N*N
        nonzero_positions = torch.randperm(N*N)[:num_nonzero]  # Random positions for non-zero elements
        nonzero_values = torch.randint(1, 10, (num_nonzero,))  # Random non-zero values between 1-9
        grid.view(-1)[nonzero_positions] = nonzero_values
        
        # Convert input grid to sequence with row markers
        pos = 0
        for row in range(N):
            X[i, pos:pos+N] = grid[row]  # Add row values
            X[i, pos+N] = 10  # Add row marker
            pos += N + 1
        X[i, pos] = 11  # Add EOS token
        
        # Randomly choose between horizontal and vertical flip
        flip_type = torch.randint(0, 2, (1,)).item()  # 0 for horizontal flip, 1 for vertical flip
        
        # Rotate grid according to chosen rotation
        hflip_grid = torch.flip(grid, dims=[1])  # Flip horizontally by flipping columns
        vflip_grid = torch.flip(grid, dims=[0])  # Flip vertically by flipping rows

        # Convert all non-zero values in the flipped grids to 5
        hflip_grid = torch.where(hflip_grid > 0, torch.tensor(5, dtype=torch.long), hflip_grid)
        vflip_grid = torch.where(vflip_grid > 0, torch.tensor(5, dtype=torch.long), vflip_grid)

        if torch.equal(hflip_grid, vflip_grid):
            # Under-determined grid... don't keep it.
            continue

        # Convert flipped grid to sequence with row markers
        pos = 0
        for row in range(N):
            if flip_type == 0:
                Y[i, pos:pos+N] = hflip_grid[row]  # Add rotated row values
            else:
                Y[i, pos:pos+N] = vflip_grid[row]  # Add rotated row values
            Y[i, pos+N] = 10  # Add row marker
            pos += N + 1
        Y[i, pos] = 11  # Add EOS token
        
        if torch.equal(X, Y):
            continue

        # Concatenate X and Y for the input
        combined_input[i, :max_len] = X[i]
        combined_input[i, max_len:2*max_len] = Y[i]
        
        # Create label with SOS token
        labels[i, 0] = SOS_token
        labels[i, 1] = flip_type  # 0 for horizontal flip, 1 for vertical flip
        i += 1

    return combined_input, labels

X_train, Y_train = generate_data(trainN)
X_val, Y_val = generate_data(valN)
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
                                if token == 10:  # Row marker
                                    if current_row:  # Only append if row is not empty
                                        grid.append(current_row)
                                        current_row = []
                                elif token == 11:  # EOS token
                                    if current_row:  # Only append if row is not empty
                                        grid.append(current_row)
                                    break
                                elif token == 12:  # PAD token
                                    continue
                                else:
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

# Initialize model
model = StandardTransformerModel(d_model, 
                                 num_heads, 
                                 num_layers, 
                                 input_vocab_size, 
                                 num_classes,
                                 max_seq_length).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)

print(f"Training for {num_epochs} epochs on a total of {trainN} samples...")
# Training Loop
for idx, epoch in enumerate(range(num_epochs)):
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
        if epoch > 500:
            accuracy = evaluate(val_loader, verbose=True)
        else:
            accuracy = evaluate(val_loader)
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}, Validation Accuracy: {accuracy:.2f}%")
