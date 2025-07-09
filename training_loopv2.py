import numpy as np 

import wandb

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset_class import CMBdataset
from CosmicNetwork import CosmicNetwork, CosmicNetwork_v2

from tqdm import tqdm
from pathlib import Path


# ───────────────────────────────────────────────────────────────
#  W&B sweep config  ➜  cfg.<param>
# ───────────────────────────────────────────────────────────────
wandb.init(
    project="hyper_parameter_cmb",
    # name   ="run",        
    config={
        "epochs":        100,
        "batch_size":    16,
        "learning_rate": 2e-3,
        "hidden_dim":    128,
        "hidden_layers": 3,
        "patience":      12
    }
)
cfg = wandb.config
run_name = wandb.run.name 

WEIGHT_DIR = Path("hyper_weights")
WEIGHT_DIR.mkdir(exist_ok=True)

# parameters
h5_path             = 'cmb_dataset.h5'
ell_slice           = slice(2, 801)
split               = [0.8, 0.2]
batch_size          = 32    
num_workers         = 8


# ───────────────────────────────────────────────────────────────
#  Data loaders use sweep batch_size
# ───────────────────────────────────────────────────────────────
def build_dataset(file_path, ell_slice,
                  split, batch_size=32,
                  num_workers=4, pin_memory=True, seed=42):
    """
    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    """
    # 1.  Load full dataset  ------------------------------------
    ds = CMBdataset(Path(file_path), ell_slice)

    # 2.  Deterministic random split ----------------------------
    n_train = int(split[0] * len(ds))
    n_val   = len(ds) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=generator)

    # 3.  Build loaders  ----------------------------------------
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,          # shuffle only training
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    val_loader   = DataLoader(val_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    return train_loader, val_loader

train_loader, val_loader = build_dataset(
    h5_path, ell_slice, split,
    batch_size=cfg.batch_size,
    num_workers=num_workers
)

# ───────────────────────────────────────────────────────────────
#  Model / optimiser / scheduler
# ───────────────────────────────────────────────────────────────
model = CosmicNetwork_v2(
            in_dim=2,
            out_dim=799,
            hidden_dim=cfg.hidden_dim,
            hidden_layers=cfg.hidden_layers)
wandb.watch(model, log="all")

loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
               optimizer, mode="min",
               factor=0.1, patience=cfg.patience - 4,
               min_lr=1e-6)

best_val   = float("inf")
epochs_no_improve = 0

# ───────────────────────────────────────────────────────────────
#  Training loop
# ───────────────────────────────────────────────────────────────
for epoch in tqdm(range(cfg.epochs), desc="Training"):

    # ---- Train -------------------------------------------------
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        pred  = model(X)
        loss  = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ---- Validate ---------------------------------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xv, yv in val_loader:
            val_loss += loss_fn(model(Xv), yv).item()
    val_loss /= len(val_loader)

    # ---- LR schedule & logging --------------------------------
    scheduler.step(val_loss)
    lr_now = optimizer.param_groups[0]["lr"]
    wandb.log({
        "epoch":      epoch,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "lr":         lr_now
    })

    # ---- Early stopping + best-model checkpoint -------------------
    ckpt_path = WEIGHT_DIR / f"weights-{run_name}.pt"
    
    if val_loss < best_val - 1e-4:          # new best
        best_val = val_loss
        epochs_no_improve = 0
    
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "opt_state":   optimizer.state_dict(),
            "val_loss":    val_loss
        }, ckpt_path)
    
        # upload to the run's Files tab
        wandb.save(str(ckpt_path))
    
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        