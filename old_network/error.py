import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn

# -------------------------
# 1. Load Dataset
# -------------------------

df = pd.read_csv("formatted_better_noodles.csv")
omega_b_vals = df["omega_b"].values[2:]
Cl_TT_vals = df.drop(columns=["omega_b"]).values[2:]

# Sort by omega_b for smooth animation
sorted_indices = np.argsort(omega_b_vals)
omega_b_vals_sorted = omega_b_vals[sorted_indices]
Cl_TT_sorted = Cl_TT_vals[sorted_indices]

# Choose 60 evenly spaced frames
n_frames = 2998
frame_indices = np.linspace(0, len(omega_b_vals_sorted) - 1, n_frames, dtype=int)
omega_b_anim = omega_b_vals_sorted[frame_indices]
Cl_TT_truth = Cl_TT_sorted[frame_indices]

# -------------------------
# 2. Load Model
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
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Generate predictions for selected omega_b
omega_tensor = torch.tensor(omega_b_anim, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    Cl_TT_pred = model(omega_tensor).numpy()
    
    
import time

# Choose one omega_b value (e.g., the first one)
omega_sample = torch.tensor([[omega_b_anim[0]]], dtype=torch.float32)

# Warm-up (optional, useful for avoiding lazy init effects)
for _ in range(10):
    _ = model(omega_sample)

# Timing
start_time = time.time()
with torch.no_grad():
    output = model(omega_sample)
end_time = time.time()

print(f"Single inference time: {(end_time - start_time)*1000:.4f} ms")

