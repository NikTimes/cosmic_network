import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -------------------------
# 1. Load & Prepare Dataset
# -------------------------

# torch.set_printoptions(precision=8)

# Load dataset
dataset_df = pd.read_csv("formatted_better_noodles.csv")

test_index = np.random.randint(0, 2998, size=200)
all_indices = np.arange(len(dataset_df) - 2)
test_mask = np.isin(all_indices, test_index)

# Split into input (omega_b) and output (Cl_TT)
X_full = torch.tensor(dataset_df["omega_b"].values[2:], dtype=torch.float32).view(-1, 1)
Y_full = torch.tensor(dataset_df.drop(columns=["omega_b"]).values[2:], dtype=torch.float32)

# Training Set
X_train = X_full[~test_mask]
Y_train = Y_full[~test_mask]

# Test Set 
X_test = X_full[test_mask]
Y_test =Y_full[test_mask]


# final datasets 

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------
# 2. Define Neural Network
# -------------------------

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2501)
        )

    def forward(self, x):
        return self.model(x)

model = MyNet()

# -------------------------
# 3. Training Setup
# -------------------------

loss_fn = nn.L1Loss()  # or try SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 1000

# -------------------------
# 4. Training Loop
# -------------------------

for epoch in tqdm(range(n_epochs), desc="Training"):
    epoch_loss = 0.0
    num_batches = 0

    for X_batch, Y_batch in train_loader:
        pred = model(X_batch)
        loss = loss_fn(pred, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches

# Save model weights after training
torch.save(model.state_dict(), "trained_model.pth")

# -------------------------
# 5. Test and error Calculation 
# -------------------------

model.eval()
total_error = 0.0
count = 0
epsilon = 1e-8  # to avoid division by zero

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        preds = model(X_batch)
        mask = Y_batch > 1.0  # Only compare where C_ell is meaningful
        
        # Compute normalized absolute percentage error per batch
        abs_diff = torch.abs(preds - Y_batch)
        norm_diff = abs_diff / (torch.abs(Y_batch) + epsilon)
        
        batch_error = torch.sum(norm_diff).item()
        total_error += batch_error
        count += Y_batch.numel()

# Final error in percentage
avg_percentage_error = (total_error / count) * 100
print(f"Test Set Average Percentage Error: {avg_percentage_error:.2f}%")


