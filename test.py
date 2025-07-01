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

# -------------------------
# 3. Set up the animation
# -------------------------

fig, ax = plt.subplots(figsize=(10, 5))
line_truth, = ax.plot([], [], lw=2, label="True Cl_TT")
line_pred, = ax.plot([], [], lw=2, linestyle='--', label="Predicted Cl_TT")
ax.set_xlim(0, 2500)
ax.set_ylim(0, 7000)
ax.set_xlabel("Multipole ℓ")
ax.set_ylabel("C_ℓ")
title = ax.set_title("")
ax.grid(True)
ax.legend()

def init():
    x = np.arange(2501)
    line_truth.set_data(x, Cl_TT_truth[0])
    line_pred.set_data(x, Cl_TT_pred[0])
    title.set_text(f"omega_b = {omega_b_anim[0]:.5f}")
    return line_truth, line_pred, title

def update(frame):
    x = np.arange(2501)
    line_truth.set_data(x, Cl_TT_truth[frame])
    line_pred.set_data(x, Cl_TT_pred[frame])
    title.set_text(f"omega_b = {omega_b_anim[frame]:.5f}")
    return line_truth, line_pred, title

anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True)

# Save or show
anim.save("omega_variation_comparison.mp4", fps=200, extra_args=['-vcodec', 'libx264'])
plt.show()

