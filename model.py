import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from classy import Class

# -------------------------
# 1. Load & Prepare Dataset
# -------------------------

# Load dataset
dataset_df = pd.read_csv("cmb_dataset.csv")

# Split into input (omega_b) and output (Cl_TT)
X = torch.tensor(dataset_df["omega_b"].values[2:], dtype=torch.float32).view(-1, 1)
Y = torch.tensor(dataset_df.drop(columns=["omega_b"]).values[2:], dtype=torch.float32)

# Normalize input (recommended for stability)
X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std

# Normalize Y
Y_mean = Y.mean()
Y_std = Y.std()
Y_norm = (Y - Y_mean) / Y_std

# Use normalized Y for training
dataset = TensorDataset(X_norm, Y_norm)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 100

# -------------------------
# 4. Training Loop
# -------------------------

for epoch in tqdm(range(n_epochs), desc="Training"):
    epoch_loss = 0.0
    num_batches = 0

    for X_batch, Y_batch in loader:
        pred = model(X_batch)
        loss = loss_fn(pred, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    # if epoch % 100 == 0:
        # print(f"Epoch {epoch:4d} | Avg Loss: {avg_loss:.6e}")

# -------------------------
# 5. Test Model on New Input
# -------------------------

omega_b_test = 0.022
omega_b_tensor = torch.tensor([[omega_b_test]], dtype=torch.float32)
omega_b_tensor_norm = (omega_b_tensor - X_mean) / X_std

model.eval()
with torch.no_grad():
    y_pred_norm = model(omega_b_tensor_norm).squeeze().numpy()  # Get normalized prediction
    y_pred = y_pred_norm * Y_std.numpy() + Y_mean.numpy()       # Denormalize prediction


# -------------------------
# 6. Compare with CLASS (Unlensed, raw Cls)
# -------------------------

cosmo = Class()
cosmo.set({
    "omega_b": omega_b_test,
    "omega_cdm": 0.1201075,
    "h": 0.67810,
    "A_s": 2.100549e-09,
    "n_s": 0.9660499,
    "tau_reio": 0.05430842,
    "output": "tCl",
    "l_max_scalars": 2500
})
cosmo.compute()
cls = cosmo.raw_cl(2500)['tt']
cosmo.struct_cleanup()
cosmo.empty()

# -------------------------
# 7. Plot Prediction vs CLASS (D_ell)
# -------------------------

ell = np.arange(len(y_pred))
D_ell_pred = (ell * (ell + 1) * y_pred) / (2 * np.pi)
D_ell_cls = (ell * (ell + 1) * cls) / (2 * np.pi)

plt.plot(ell, D_ell_cls, label='CLASS (raw Cl)', linewidth=1)
plt.plot(ell, D_ell_pred, '--', label='NN Prediction', linewidth=1)
plt.xlabel("Multipole ℓ")
plt.ylabel(r"$D_\ell^{TT} = \ell(\ell+1)C_\ell^{TT}/2\pi$")
plt.title(f"Prediction vs CLASS for ω_b = {omega_b_test}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 8. Compute % Error
# -------------------------

percent_error = np.mean(np.abs((D_ell_pred - D_ell_cls) / D_ell_cls)) * 100
print(f"\n✅ Avg % error vs CLASS (D_ell): {percent_error:.3f}%")
