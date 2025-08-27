from classy import Class
import numpy as np, h5py 
from tqdm import tqdm
from smt.sampling_methods import LHS
import torch



def build_dataset(
        omega_b_range   = [0.005, 0.040],
        omega_cdm_range = [0.001, 0.99],
        l_max           = 2500,
        num_points      = 10_000,        
        out_path        = "cmb_dataset.h5"
        ):
    
    x_limits = np.array([
        omega_b_range,
        omega_cdm_range
        ], float)
    
    sampler = LHS(xlimits=x_limits)
    samples = sampler(num_points)
    
    ells    = np.arange(l_max + 1)
    num_ell = len(ells)
    
    with h5py.File(out_path,"w") as h5:
        
        d_omega_b       = h5.create_dataset("omega_b", shape=(num_points,), dtype="f4")
        d_omega_cdm     = h5.create_dataset("omega_cdm", shape=(num_points,), dtype="f4")
        d_omega_lambda  = h5.create_dataset("omega_lambda", shape=(num_points,), dtype="f4")
        d_logClTT       = h5.create_dataset("log_C_ell", shape=(num_points, num_ell),
                                           dtype="f4", compression="gzip", chunks=True)
        d_ell           = h5.create_dataset("ell", data=ells, dtype="i4")
        
        for i, (omega_b, omega_cdm) in enumerate(tqdm(samples, desc="Generating C_ell")):
            
            omega_b     = float(omega_b)
            omega_cdm   = float(omega_cdm)
            omega_lam   = 1 - omega_b - omega_cdm
            
            if omega_lam < 0:
                print(f"Skipping unphysical sample {i}: Ω_b={omega_b:.4f}, Ω_cdm={omega_cdm:.4f}, Ω_Λ={omega_lam:.4f}")
                continue
            
            bbn_path = "/home/enric/Repositories/class_public/external/bbn/sBBN_2025.dat"
            
            parameters  = {
                
                "omega_b"       : omega_b, 
                "omega_cdm"     : omega_cdm, 
                # "Omega_lambda"  : omega_lam,
                "Omega_k"       : 0.0,
                "h"             : 0.67810,
                "A_s"           : 2.1e-9,
                "n_s"           : 0.966,
                "tau_reio"      : 0.054,
                "lensing"       : "no",
                "output"        : "tCl",
                "l_max_scalars" : l_max,
                "YHe"           : 0.2471      
                }
            
            try:
                cosmo           = Class()
                cosmo.set(parameters)
                cosmo.compute()
                
                cls             = cosmo.raw_cl(l_max)
                cl_tt           = cls['tt']
                log_C_ell       = np.log(cl_tt)
                
                d_omega_b[i]        = omega_b
                d_omega_cdm[i]      = omega_cdm
                d_omega_lambda[i]   = omega_lam
                d_logClTT[i, :]     = log_C_ell
            
            except Exception as e:
                print(f"Error at sample {i} (Ω_b={omega_b}, Ω_cdm={omega_cdm}): {e}")
                
            finally:
                cosmo.struct_cleanup()
                cosmo.empty()
        
        print(f"Dataset written to: {out_path}")
        
        
class CMBdataset(torch.utils.data.Dataset):
    
    def __init__(self, h5_path, ell_slice=None, d_ell_val=True):
        
        self.h5_path    = h5_path
        self.ell_slice  = ell_slice  # optional: ℓ=2 to 800
        self._h5        = None
        self._ell       = None
        self.d_ell_val  = d_ell_val
        
    @property
    def h5(self):
        
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        
        return self._h5
            
    @property
    def ell(self):
        if self._ell is None:
            self._ell = self.h5["ell"][:]        # copy once into RAM
        return self._ell
    
    
    def __len__(self):
        return len(self.h5["omega_b"])
    
    def __getitem__(self, index):
       
       omega_b      = self.h5["omega_b"][index]
       omega_cdm    = self.h5["omega_cdm"][index]
       omega_lambda = self.h5["omega_lambda"][index]

       log_cl       = self.h5["log_C_ell"][index]
       ell          = self.ell
       
       if self.ell_slice is not None:
            ell    = ell[self.ell_slice]
            log_cl = log_cl[self.ell_slice]
            
       d_ell       = ell * (ell + 1.0) * np.exp(log_cl) / (2*np.pi)
           
       x           = torch.tensor([omega_b, omega_cdm], dtype=torch.float32)  
       
       if self.d_ell_val: 
           y       = torch.tensor(d_ell, dtype=torch.float32)
       else:
           y       = torch.tensor(log_cl, dtype=torch.float32)

         
       return x, y


class CMBdatasetV2(torch.utils.data.Dataset):
    def __init__(self, h5_path, ell_slice=None,
                 d_ell_val=False, in_ram=True):
        """
        Parameters
        ----------
        h5_path : str | Path
        ell_slice : slice or None   # e.g. slice(2, 801)
        d_ell_val : bool            # True → return ℓ(ℓ+1)Cℓ/2π
        in_ram : bool               # True → materialise slice once
        """
        self.h5_path    = str(h5_path)
        self.ell_slice  = ell_slice
        self.d_ell_val  = d_ell_val
        self.in_ram     = in_ram

        self._h5 = None             # lazily opened per worker
        self._ell = None

        # ── RAM materialisation ───────────────────────────────────
        if in_ram:
            with h5py.File(self.h5_path, "r") as f:
                # parameters
                self._omega = torch.stack(
                    [ torch.from_numpy(f["omega_b"][:]),
                      torch.from_numpy(f["omega_cdm"][:]) ],
                    dim=1).float()

                # ell axis
                ell = f["ell"][:]
                if ell_slice is not None:
                    ell = ell[ell_slice]
                self._ell = torch.from_numpy(ell).float()

                # log C_ℓ slice
                self._logCl = torch.from_numpy(
                    f["log_C_ell"][:, ell_slice][...]
                ).float()

    # -----------------------------------------------------------------
    def _require_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def _require_ell(self):
        if self._ell is None:
            ell = self._require_h5()["ell"][:]
            if self.ell_slice is not None:
                ell = ell[self.ell_slice]
            self._ell = torch.from_numpy(ell).float()
        return self._ell

    # -----------------------------------------------------------------
    def __len__(self):
        if self.in_ram:
            return self._omega.shape[0]
        return len(self._require_h5()["omega_b"])

    def __getitem__(self, idx):
        if self.in_ram:
            x = self._omega[idx]
            log_cl = self._logCl[idx]
            ell = self._ell
        else:
            h5 = self._require_h5()
            x = torch.tensor([h5["omega_b"][idx],
                              h5["omega_cdm"][idx]], dtype=torch.float32)
            log_cl = torch.from_numpy(
                h5["log_C_ell"][idx, self.ell_slice]).float()
            ell = self._require_ell()

        if self.d_ell_val:
            y = ell * (ell + 1.) * torch.exp(log_cl) / (2*np.pi)
        else:
            y = log_cl
        return x, y
    

class CMBdatasetPCA(torch.utils.data.Dataset):
    """
    Dataset for PCA–compressed C_ℓ data.
    Each sample:
        x  – tensor (n_parameters,)      cosmological parameters
        y  – tensor (n_components,)      PCA coefficients
    The optional arguments ell_slice and d_ell_val are accepted so existing
    training code does not break, but they are not used (PCA coeffs already
    contain the ℓ information).
    """
    # -------------------------------------------------------------
    def __init__(self, h5_path, ell_slice=None,
                 d_ell_val=False, in_ram=True):
        self.h5_path   = str(h5_path)
        self.in_ram    = in_ram

        self._h5       = None            # lazy handle for mmap mode
        self.param_names = None          # filled below
        
        

        if in_ram:
            with h5py.File(self.h5_path, "r") as f:
                # discover parameter list from file attributes
                self.param_names = f.attrs["param_names"].split(",")

                # load everything into RAM once
                self._params = torch.stack(
                    [torch.from_numpy(f[name][:])
                     for name in self.param_names],
                    dim=1).float()                          # (N, n_params)

                self._coeff  = torch.from_numpy(
                    f["coefficients"][:]).float()          # (N, n_components)
        # store optional values but ignore them
        self.ell_slice  = ell_slice
        self.d_ell_val  = d_ell_val

    # -------------------------------------------------------------
    # lazy helpers for mmap mode ----------------------------------
    def _require_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            self.param_names = self._h5.attrs["param_names"].split(",")
        return self._h5

    # -------------------------------------------------------------
    def __len__(self):
        if self.in_ram:
            return self._params.shape[0]
        return len(self._require_h5()[self.param_names[0]])

    def __getitem__(self, idx):
        if self.in_ram:
            x = self._params[idx]        # parameters
            y = self._coeff[idx]         # PCA coefficients
        else:
            h5 = self._require_h5()
            x = torch.tensor([h5[name][idx] for name in self.param_names],
                             dtype=torch.float32)
            y = torch.from_numpy(h5["coefficients"][idx]).float()
        return x, y

    # -------------------------------------------------------------
    # optional utilities -----------------------------------------
    def reconstruct_logCl(self, coeff_batch):
        """
        Reconstruct log‑C_ℓ curves from a batch of PCA coefficients.
        coeff_batch : tensor (..., n_components)
        returns      : tensor (..., n_ell)
        """
        h5 = self._require_h5() if not self.in_ram else None
        basis = (torch.from_numpy(h5["basis"][:]).float()
                 if h5 else self._basis)  # (_basis filled only if needed)
        mean  = (torch.from_numpy(h5["mean_spectrum"][:]).float()
                 if h5 else self._mean)

        return coeff_batch @ basis + mean

