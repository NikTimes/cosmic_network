import numpy as np
from classy import Class
import pandas as pd
from tqdm import tqdm


def generate_cmb_dataset(omega_b_range=(0.010, 0.030), num_points=300,
                         l_max=2500, output_file="cmb_dataset.csv"):
    omega_b_vals = np.linspace(*omega_b_range, num_points)

    base_params = {
        "omega_cdm": 0.1201075,
        "h": 0.67810,
        "A_s": 2.100549e-09,
        "n_s": 0.9660499,
        "tau_reio": 0.05430842,
        "output": "tCl",
        "lensing": "no",
        "l_max_scalars": l_max,
    }

    T_CMB_uK = 2.7255e6  # μK

    data = []
    ells = np.arange(0, l_max + 1)
    ell_factor = ells * (ells + 1) / (2 * np.pi)

    for omega_b in tqdm(omega_b_vals, desc="Generating D_ℓ^TT"):
        cosmo = Class()
        cosmo.set(base_params)
        cosmo.set({"omega_b": omega_b})
        cosmo.compute()
        cls = cosmo.raw_cl(l_max)
        cl_tt = cls['tt']
        d_ell = ell_factor[:len(cl_tt)] * cl_tt * T_CMB_uK**2
        data.append(d_ell)
        cosmo.struct_cleanup()
        cosmo.empty()

    df = pd.DataFrame(data)
    df.insert(0, "omega_b", omega_b_vals)
    df.to_csv(output_file, index=False)

    return df


"""
# generate_cmb_dataset(num_points=3000, output_file="bbbetter_noodles.csv")
# Load the dataset
df = pd.read_csv("better_noodles.csv")

# Rename the first column to 'omega_b'
df.columns.values[0] = "omega_b"

# Optional: Rename the rest of the columns to 'ell_0', 'ell_1', ..., 'ell_2500'
df.columns = ["omega_b"] + [f"ell_{i}" for i in range(1, df.shape[1])]

# Save to a new CSV file (optional)
df.to_csv("formatted_better_noodles.csv", index=False)

# Preview the first few rows
"""
