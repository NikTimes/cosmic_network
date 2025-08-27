import numpy as np 
import wandb
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset_class import CMBdatasetPCA
from CosmicNetwork import CosmicNetwork_v2

from tqdm import tqdm
from pathlib import Path
import h5py   # NEW

# ───────────────────────────────────────────────
#  W&B sweep config  ➜  cfg.<param>
# ───────────────────────────────────────────────
wandb.init(
    project="hyper_parameter_cmb_final",      
    config={
        "epochs":        200,
        "batch_size":    32,
        "learning_rate": 0.0016,
        "hidden_dim":    192,
        "hidden_layers": 4,
        "patience":      20
    }
)

cfg = wandb.config
run_name = wandb.run.name 

WEIGHT_DIR = Path("hyper_weights_9")
WEIGHT_DIR.mkdir(exist_ok=True)

# Hyperparameters 
h5_path             = "dataset/merged_complete_50_pca.h5"
ell_slice           = slice(2, 1000)
split               = [0.91, 0.09]
batch_size          = 32    
num_workers         = 8


# ───────────────────────────────────────────────
#  Dataloading
# ───────────────────────────────────────────────
def build_dataset(file_path, ell_slice, split,
                  batch_size=32, num_workers=4,
                  pin_memory=True, seed=42): 
    
    ds = CMBdatasetPCA(Path(file_path), ell_slice, d_ell_val=False)
    
    n_train = int(split[0] * len(ds))
    n_val   = len(ds) - n_train
    generator = torch.Generator().manual_seed(seed) 
    
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, test_loader

train_loader, test_loader = build_dataset(h5_path, ell_slice, split,
                                          batch_size=cfg.batch_size,
                                          num_workers=num_workers)


# ───────────────────────────────────────────────
#  Load PCA info (basis, mean, eigenvalues, ℓ)
# ───────────────────────────────────────────────
with h5py.File(h5_path, "r") as f:
    PCA_BASIS = torch.tensor(f["basis"][:], dtype=torch.float32)          # [n_comp, n_ell]
    PCA_MEAN  = torch.tensor(f["mean_spectrum"][:], dtype=torch.float32)  # [n_ell]
    ELL       = torch.tensor(f["ell"][:], dtype=torch.float32)
    VARS      = torch.tensor(f["explained_variance"][:], dtype=torch.float32)

# Standardization scale for PCA coefficients (per-component std = sqrt(var))
STD = torch.sqrt(VARS)    # [n_comp]

# ───────────────────────────────────────────────
#  Loss: CosmoPower-style (MSE on standardized coeffs)
# ───────────────────────────────────────────────
def coeff_mse_loss(y_pred_coeffs, y_true_coeffs, std):
    """
    Predict standardized PCA coefficients.
    Target = (true_coeffs / std).
    Loss   = MSE(pred, target).
    """
    y_true_std = y_true_coeffs / std
    return F.mse_loss(y_pred_coeffs, y_true_std)

# (Optional) Reconstruction helper for evaluation/plots (not used in training)
def pca_reconstruct(std_coeffs, basis, mean, std):
    """
    std_coeffs: [B, n_comp] standardized PCA coeffs (model outputs).
    returns:    [B, n_ell]  log-spectrum (log D_ell or log C_ell depending on PCA build).
    """
    coeffs = std_coeffs * std
    return coeffs @ basis + mean


# ───────────────────────────────────────────────
#  Model, optimizer, scheduler
# ───────────────────────────────────────────────
# Make sure out_dim matches the number of PCA components in the H5 file.
model = CosmicNetwork_v2(in_dim=4,
                         out_dim=PCA_BASIS.shape[0],   # <- auto-match components
                         hidden_dim=cfg.hidden_dim,
                         hidden_layers=cfg.hidden_layers)

wandb.watch(model, log='all')

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=3e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=3e-5)
GRAD_CLIP = 1.0

best_val = float("inf")
epochs_no_improve = 0


# ───────────────────────────────────────────────
#  Training Loop 
# ───────────────────────────────────────────────
for epoch in tqdm(range(cfg.epochs), desc="Training"):
    
    model.train()
    train_loss = 0.0
    
    for X, y in train_loader:
        pred = model(X)  # predicts standardized coeffs
        loss = coeff_mse_loss(pred, y, STD.to(X.device))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation ------------------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            loss = coeff_mse_loss(pred, y, STD.to(X.device))
            val_loss += loss.item()
    val_loss /= len(test_loader)

    # Scheduler + logs
    scheduler.step()
    lr_now = optimizer.param_groups[0]["lr"]
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})
    
    # Early stopping
    weights_path = WEIGHT_DIR / f"weights-{run_name}.pt"
    if val_loss < best_val - 1e-4:          
        best_val = val_loss
        epochs_no_improve = 0
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "opt_state":   optimizer.state_dict(),
            "val_loss":    val_loss
        }, weights_path)
        wandb.save(str(weights_path))
    else: 
        epochs_no_improve += 1
        if epochs_no_improve >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
