from classy import Class
import numpy as np, h5py 
from tqdm import tqdm
from smt.sampling_methods import LHS
import torch



def build_dataset(
        omega_b_range   = [0.001, 0.040],
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
    
    ells = np.arange(l_max + 1)
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
                
                "Omega_b"       : omega_b, 
                "Omega_cdm"     : omega_cdm, 
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
    
    def __init__(self, h5_path, ell_slice=None):
        
        self.h5_path = h5_path
        self.ell_slice = ell_slice  # optional: ℓ=2 to 800
        self._h5 = None
        
    @property
    def h5(self):
        
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        
        return self._h5
            
    def __len__(self):
        return len(self.h5["omega_b"])
    
    def __getitem__(self, index):
       
       omega_b      = self.h5["omega_b"][index]
       omega_cdm    = self.h5["omega_cdm"][index]
       omega_lambda = self.h5["omega_lambda"][index]

      
       log_cl = self.h5["log_C_ll"][index]
       if self.ell_slice:
           log_cl = log_cl[self.ell_slice]
           
       x = torch.tensor([omega_b, omega_cdm, omega_lambda], dtype=torch.float32)  
       y = torch.tensor(log_cl, dtype=torch.float32)

       return x, y
   
build_dataset()
