import h5py, numpy as np, pathlib#
from sklearn.decomposition import PCA

import os
from pathlib import Path
import shutil
import numpy as np



param_names = ['omega_b', 'omega_c', 'omega_g', 'omega_ur']
   

def clean_dataset(in_path, out_path, param_names, ell_keep=999):
    with h5py.File(in_path, 'r') as fin, h5py.File(out_path, 'w') as fout:
        mask = np.logical_or.reduce([(fin[p][:] != 0) for p in param_names])
        idx  = np.where(mask)[0]

        for name, dset in fin.items():
            data = dset[:]

            if name == "d_ell":
                data = data[:, :ell_keep]   # 🔥 truncate spectra
            if name == "ell":
                data = dset[:ell_keep]      # 🔥 keep first 999 ℓ values

            if data.shape[0] == fin[param_names[0]].shape[0]:
                fout.create_dataset(name, data=data[idx], compression='gzip', compression_opts=6)
            else:
                fout.create_dataset(name, data=data)

    print(f"Wrote {len(idx)} rows →", out_path)

    

    
def check_ranges(path, param_names):
    """Check min, max, mean of omega parameters in the given dataset."""
    with h5py.File(path, 'r') as f:
        print(f"\n🔍 Checking ranges in: {os.path.basename(path)}") 
        for name in param_names:
            if name in f:
                data = f[name][:]
                print(f"  • {name}: min = {np.min(data):.4g}, max = {np.max(data):.4g}, mean = {np.mean(data):.4g}")
            else:
                print(f"  • {name} missing!")


def check_dataset(path):
    
    with h5py.File(path, 'r') as f:
        
        print("Length of samples:", len(f['omega_b']))            # or f['omega_b'][:]
        print("d_ell shape:", f['d_ell'].shape)           # shape of 2D array (samples × ℓ)
        print("ell shape:", f['ell'].shape)                       # shape of ℓ array
        print("First 5 omega_b values:", f['omega_b'][:5])
        print("First d_ell row:", f['d_ell'][0, :5])
        

def merge_dataset(file1, file2, output_file):
    
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2, h5py.File(output_file, 'w') as fout:
        
        for key in f1.keys():
            
            d1 = f1[key][:]
            d2 = f2[key][:]
            
            # Skip for ell
            if key == 'ell':
                fout.create_dataset(key, data=d1)
                print(f"✔ Copied '{key}' without merging: shape {d1.shape}")
                continue
            
            merged = np.concatenate([d1, d2], axis = 0)
            fout.create_dataset(key, data=merged)
            print(f"✔ Merged '{key}': {d1.shape} + {d2.shape} → {merged.shape}")  
            

def pca_component(in_path, out_path, n_components, param_names, ell_slice=None):
    with h5py.File(in_path, 'r') as fin, h5py.File(out_path, 'w') as fout:
        
        # Load log(C_ell) spectra
        log_cl = fin['d_ell'][:]

        # Extract and save each parameter by name
        for name in param_names:
            param_data = fin[name][:]
            fout.create_dataset(name, data=param_data)

        # Stack for PCA input (only used internally)
        parameter_list = [fin[name][:] for name in param_names]
        parameters = np.stack(parameter_list, axis=1)

        # Perform PCA
        pca = PCA(n_components=n_components)
        alpha = pca.fit_transform(log_cl)
        basis = pca.components_
        mean_spectrum = pca.mean_

        # Save PCA output
        fout.create_dataset('coefficients', data=alpha)
        fout.create_dataset('basis', data=basis)
        fout.create_dataset('mean_spectrum', data=mean_spectrum)
        fout.create_dataset('explained_variance', data=pca.explained_variance_)  # NEW

        # Save ℓ values (either from slice or inferred from spectrum length)
        if ell_slice is not None:
            ell = np.arange(ell_slice.start, ell_slice.stop)
        else:
            ell = np.arange(log_cl.shape[1])
        fout.create_dataset('ell', data=ell)  # NEW

        # Metadata
        fout.attrs['n_components'] = n_components
        fout.attrs['n_samples'] = log_cl.shape[0]
        fout.attrs['n_ell'] = log_cl.shape[1]
        fout.attrs['n_parameters'] = len(param_names)
        fout.attrs['param_names'] = ','.join(param_names)

    print(f"PCA complete and saved to {out_path}")

def check_pca_dataset(path):
    with h5py.File(path, 'r') as f:
        print("Datasets in file:")
        for name in f.keys():
            print(f"  • {name}: shape = {f[name].shape}")

        print("\nSample data preview:")
        if 'coefficients' in f:
            print("  First 5 PCA coefficients:")
            print(f['coefficients'][:5])

        if 'basis' in f and 'mean_spectrum' in f:
            print("\nPCA basis shape:", f['basis'].shape)
            print("Mean spectrum shape:", f['mean_spectrum'].shape)

        param_names = f.attrs.get('param_names', '').split(',')
        for name in param_names:
            if name in f:
                print(f"\n  First 5 values of {name}:")
                print(f[name][:5])

        print("\nFile attributes:")
        for key in f.attrs:
            print(f"  {key}: {f.attrs[key]}")



