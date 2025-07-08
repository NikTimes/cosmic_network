import numpy as np 

import torch 
from torch.nn import nn
from torch.utils.data import DataLoader, random_split

from dataset_class import CMBdataset
from CosmicNetwork import CosmicNetwork

from tqdm import tqdm
from pathlib import Path

# --------------------------------------------
# Load File initiate dataset 
# --------------------------------------------

# parameters
h5_path             = 'cmb_dataset.h5'
ell_slice           = slice(2, 801)
split               = [0.8, 0.2]

def build_dataset(file_path, ell_slice, split, batch_size=32, num_workers=4):
    
    h5_file          = Path(file_path)
    ell_range        = ell_slice
    
    ds               = CMBdataset(h5_file, ell_range)
    
    total   = len(ds)
    n_train = int(split[0] * total)
    n_val   = total - n_train

    train_ds, val_ds = random_split(ds, [n_train, n_val])
    
    train_loader     = DataLoader(train_ds, shuffle=True, 
                                  batch_size=batch_size, 
                                  num_workers=num_workers)
 
    val_loader       = DataLoader(val_ds, shuffle=False,
                                  batch_size=batch_size,
                                  num_workers=num_workers)
    
    
    X_train, Y_train = zip(*[train_ds[i] for i in range(len(train_ds))])
    X_val,   Y_val   = zip(*[val_ds[i]   for i in range(len(val_ds))])
    
    X_train          = torch.stack(X_train)
    Y_train          = torch.stack(Y_train)
    X_val            = torch.stack(X_val)
    Y_val            = torch.stack(Y_val)
    
    return train_loader, val_loader, X_train, Y_train, X_test, Y_test    


train_loader, val_loader, X_train, Y_train, X_test, Y_test = build_dataset(h5_path, 
                                                                   ell_slice, 
                                                                   split) 

# --------------------------------------------
# Training Loop
# --------------------------------------------

model       = CosmicNetwork()

# parameters

loss_fn         = nn.MSELoss()
learning_rate   = 1e-3
optimizer       = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs        = 100   


# loop
for epoch in tqdm(range(n_epochs), desc="Training"):
    
    epoch_loss  = 0.0
    num_batches = 0 
    
    for X_batch, Y_batch in train_loader:
        
        pred        = model(X_batch)
        loss        = loss_fn(pred, Y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss  += loss.item()
        num_batches += 1
        
    avg_loss    = epoch_loss / num_batches 
    

    
    
    